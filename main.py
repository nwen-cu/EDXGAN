import os
from io import StringIO
import random
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchinfo
import torchview
import wandb
from omegaconf import OmegaConf
from data import datamodule
import hydra


# ---------------------------
# Networks (Model Part)
# ---------------------------
class Generator(nn.Module):
    def __init__(self, pretrained_checkpoint: str = None):
        super(Generator, self).__init__()

        if pretrained_checkpoint:
            from pretrain.vgg16_feal_preft import VGG16Feal
            state_dict = torch.load(pretrained_checkpoint)['state_dict']
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            model_pretrained = VGG16Feal(False)
            model_pretrained.load_state_dict(state_dict)
            self.net_pretrained = nn.Sequential(*list(model_pretrained.net_pretrained[:23]))
            print('Pretrained checkpoint loaded.')
        else:
            # Input size adjusted to (b, 1, 64, 64)
            self.net_pretrained = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                *list(vgg16_bn(weights=VGG16_BN_Weights).features[1:23])
            )
            print('Default pretrained checkpoint loaded')

        self.net_global_conv: nn.Module = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.net_global_fc: nn.Module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
        )

        self.net_regress: nn.Module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 2),
        )

        self.net_mid: nn.Module = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.net_fusion: nn.Module = nn.Sequential(
            nn.Unflatten(1, (256, 1, 1)),      # Reshape: (B, 256) -> (B, 256, 1, 1)
            nn.ReplicationPad2d((0, 15, 0, 15))  # Pad: replicates the single pixel to get (B, 256, 16, 16)
        )

        self.net_color: nn.Module = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        low_features = self.net_pretrained(image)
        glob_features = self.net_global_conv(low_features)
        glob_embed = self.net_global_fc(glob_features)
        labels = self.net_regress(glob_features)
        mid_features = self.net_mid(low_features)
        glob_fusion = self.net_fusion(glob_embed)
        fus_features = torch.cat([mid_features, glob_fusion], dim=1)
        color_image = self.net_color(fus_features)
        return color_image, labels


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding='same'), 
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding='same') 
        )

    def forward(self, image_bse, image_edx):
        x = torch.cat((image_bse, image_edx), dim=1)  # Concatenate along the channel dimension
        return self.model(x)


# ---------------------------
# Model (Lightning Module)
# ---------------------------
class EDXGAN(pl.LightningModule):
    def __init__(self, cfg: OmegaConf):
        super(EDXGAN, self).__init__()
        self.cfg = cfg
        
        self.automatic_optimization = False

        self.net_g = Generator(self.cfg.model.pretrained_ckpt)
        self.net_c = Critic()

        self.loss_func_mse = nn.MSELoss()

    def on_train_start(self) -> None:
        # Force pretrained model transfer to GPU
        self.net_g.to(self.device)
        self.net_c.to(self.device)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.net_g.parameters(), lr=self.cfg.hp.lr_g, betas=(0.5, 0.99), eps=1e-07)
        opt_c = torch.optim.Adam(self.net_c.parameters(), lr=self.cfg.hp.lr_c, betas=(0.5, 0.99), eps=1e-07)
        return [opt_g, opt_c], []
    
    def training_step(self, batch, batch_idx):
        img_bse, img_edx_real, label = batch

        self._training_step_c(img_bse, img_edx_real)
        self._training_step_g(img_bse, img_edx_real, label)

    def _training_step_g(self, img_bse: torch.Tensor, img_edx_real: torch.Tensor, label: torch.Tensor):
        opt_g, _ = self.optimizers()
        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()

        img_edx_fake, label_pred = self.net_g(img_bse)

        loss_adv = self.net_c(img_bse, img_edx_fake).mean()
        loss_pix = self.loss_func_mse(img_edx_fake, img_edx_real)
        loss_lbl = self.loss_func_mse(label_pred, label)

        loss_g = self.cfg.hp.w_adv * loss_adv + self.cfg.hp.w_pix * loss_pix + self.cfg.hp.w_lbl * loss_lbl

        self.log('loss_g', loss_g)
        self.log('loss_g_adv', loss_adv)
        self.log('loss_g_pix', loss_pix)
        self.log('loss_g_lbl', loss_lbl)

        self.manual_backward(loss_g)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

    def _training_step_c(self, img_bse: torch.Tensor, img_edx_real: torch.Tensor):
        _, opt_c = self.optimizers()
        self.toggle_optimizer(opt_c)
        opt_c.zero_grad()

        img_edx_fake, _ = self.net_g(img_bse)
        loss_c_real = self.net_c(img_bse, img_edx_real).mean()
        loss_c_fake = self.net_c(img_bse, img_edx_fake).mean()

        grad_penalty = self._compute_gp(img_bse, img_edx_real, img_edx_fake)
        loss_c = loss_c_fake - loss_c_real + self.cfg.hp.w_gp * grad_penalty

        self.log('loss_c', loss_c)
        self.log('loss_c_fake', loss_c_fake)
        self.log('loss_c_real', loss_c_real)
        self.log('grad_penalty', grad_penalty)

        self.manual_backward(loss_c)
        opt_c.step()
        self.untoggle_optimizer(opt_c)

    def _compute_gp(self, img_bse: torch.Tensor, img_real: torch.Tensor, img_fake: torch.Tensor) -> torch.Tensor:
        batch_size = img_real.size(0)
        epsilon = torch.rand((batch_size, 1, 1, 1), device=self.device).expand_as(img_real)
        img_interpolated = epsilon * img_real + (1 - epsilon) * img_fake
        img_interpolated.requires_grad_(True)
        logits_interpolated = self.net_c(img_bse, img_interpolated)
        grad_outputs = torch.ones_like(logits_interpolated)
        grad_interpolated = torch.autograd.grad(
            outputs=logits_interpolated,
            inputs=img_interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0].view(batch_size, -1)
        grad_norm = grad_interpolated.norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Import needed packages
        import matplotlib.pyplot as plt
        import numpy as np

        # Unpack batch: (BSE, ground truth EDX, ground truth labels)
        img_bse, img_edx_real, label_true = batch
        n_samples = min(4, img_bse.size(0))
        img_bse_vis = img_bse[:n_samples]
        img_edx_real_vis = img_edx_real[:n_samples]
        label_true_vis = label_true[:n_samples]

        # Run generator on these samples (note: using no_grad for visualization)
        with torch.no_grad():
            img_edx_fake, label_pred = self.net_g(img_bse_vis)

        # Move tensors to CPU for plotting
        img_bse_vis = img_bse_vis.detach().cpu()
        img_edx_real_vis = img_edx_real_vis.detach().cpu()
        img_edx_fake = img_edx_fake.detach().cpu()
        label_true_vis = label_true_vis.detach().cpu()
        label_pred = label_pred.detach().cpu()

        # For generated images: they come from a tanh output in [-1, 1]. Scale them to [0, 1].
        def process_generated(img):
            return (img + 1) / 2

        # Create a mono-colored image given a single-channel image.
        # We start with a white background and replace one channel.
        def create_mono_channel_image(channel_data, color):
            H, W = channel_data.shape
            rgb = np.ones((H, W, 3))
            if color == "red":
                rgb[..., 0] = channel_data
            elif color == "green":
                rgb[..., 1] = channel_data
            elif color == "blue":
                rgb[..., 2] = channel_data
            return rgb

        # Create a figure with 8 rows and 4 columns.
        fig, axes = plt.subplots(nrows=n_samples * 2, ncols=4, figsize=(16, n_samples * 4))

        for i in range(n_samples):
            # Process BSE image (grayscale)
            bse_img = img_bse_vis[i].squeeze(0).numpy()  # shape (H, W)

            # Process generated image: scale and convert to numpy
            fake_img = process_generated(img_edx_fake[i]).numpy()  # shape (3, H, W)
            # Assume ground truth edx image is already in [0, 1]
            true_img = img_edx_real_vis[i].numpy()  # shape (3, H, W)

            # Split channels for generated image (Fe, Al, O)
            fake_fe = fake_img[0, :, :]
            fake_al = fake_img[1, :, :]
            fake_o  = fake_img[2, :, :]

            # Split channels for ground truth image
            true_fe = true_img[0, :, :]
            true_al = true_img[1, :, :]
            true_o  = true_img[2, :, :]

            # Create mono-colored images with white background for generated and ground truth channels
            fake_fe_img = create_mono_channel_image(fake_fe, "red")
            fake_al_img = create_mono_channel_image(fake_al, "green")
            fake_o_img  = create_mono_channel_image(fake_o, "blue")

            true_fe_img = create_mono_channel_image(true_fe, "red")
            true_al_img = create_mono_channel_image(true_al, "green")
            true_o_img  = create_mono_channel_image(true_o, "blue")

            # --- First row for sample i: BSE, Gen_Fe, Gen_Al, Gen_O ---
            # BSE image
            ax = axes[i * 2, 0]
            ax.imshow(bse_img, cmap="gray")
            ax.axis('off')
            ax.set_title("BSE")
            # Generated Fe channel
            ax = axes[i * 2, 1]
            ax.imshow(fake_fe_img)
            ax.axis('off')
            ax.set_title("Gen Fe")
            # Generated Al channel
            ax = axes[i * 2, 2]
            ax.imshow(fake_al_img)
            ax.axis('off')
            ax.set_title("Gen Al")
            # Generated O channel
            ax = axes[i * 2, 3]
            ax.imshow(fake_o_img)
            ax.axis('off')
            ax.set_title("Gen O")

            # --- Second row for sample i: Labels, True_Fe, True_Al, True_O ---
            # In the "Labels" cell, display both predicted and ground truth label values.
            ax = axes[i * 2 + 1, 0]
            # Convert tensor values to lists for nicer printing.
            pred_val = label_pred[i].tolist()
            true_val = label_true_vis[i].tolist()
            label_text = f"Pred: {pred_val}\nTrue: {true_val}"
            ax.text(0.5, 0.5, label_text, horizontalalignment='center',
                    verticalalignment='center', fontsize=10)
            ax.axis('off')
            ax.set_title("Labels")
            # Ground truth Fe channel
            ax = axes[i * 2 + 1, 1]
            ax.imshow(true_fe_img)
            ax.axis('off')
            ax.set_title("True Fe")
            # Ground truth Al channel
            ax = axes[i * 2 + 1, 2]
            ax.imshow(true_al_img)
            ax.axis('off')
            ax.set_title("True Al")
            # Ground truth O channel
            ax = axes[i * 2 + 1, 3]
            ax.imshow(true_o_img)
            ax.axis('off')
            ax.set_title("True O")

        plt.tight_layout()
        
        # Prepare caption with epoch, batch step, and global step.
        caption = f"Epoch: {self.current_epoch}, Step: {batch_idx}, Global Step: {self.global_step}"
        # Log the figure to wandb.
        self.logger.experiment.log({"Predicted Images": wandb.Image(fig, caption=caption)})
        plt.close(fig)


# ---------------------------
# Callback for Logging Model Info to WandB
# ---------------------------
class ModelInfoCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        # Log generator summary
        wandb.log({"generator_summary": str(torchinfo.summary(
            pl_module.net_g,
            input_size=(7, 1, 64, 64),
            device=pl_module.device,
            verbose=0
        ))})

        # Log critic summary
        wandb.log({"critic_summary": str(torchinfo.summary(
            pl_module.net_c,
            input_size=[(7, 1, 64, 64), (7, 3, 64, 64)],
            device=pl_module.device,
            verbose=0
        ))})

        # Log generator graph
        gen_graph = torchview.draw_graph(pl_module.net_g, input_size=(7, 1, 64, 64), device='cpu')
        gen_graph.visual_graph.render(filename="generator_graph", format="png", cleanup=True)
        wandb.log({"generator_graph": wandb.Image("generator_graph.png")})

        # Log critic graph
        crit_graph = torchview.draw_graph(pl_module.net_c, input_size=[(7, 1, 64, 64), (7, 3, 64, 64)], device='cpu')
        crit_graph.visual_graph.render(filename="critic_graph", format="png", cleanup=True)
        wandb.log({"critic_graph": wandb.Image("critic_graph.png")})


class RandomRotate:
    """
    Randomly rotates a PIL image by one of the specified angles.

    Parameters
    ----------
    angles : list of int, optional
        List of angles (in degrees) to choose from for rotation. 
        Default is [0, 90, 180, 270].

    Methods
    -------
    __call__(img)
        Rotates the given PIL image by a randomly selected angle from the list.
    """
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Rotate the input PIL image by a random angle from the provided list.

        Parameters
        ----------
        img : PIL.Image.Image
            Input image to be rotated.

        Returns
        -------
        PIL.Image.Image
            Rotated image.
        """
        angle = random.choice(self.angles)
        return img.rotate(angle)


# ---------------------------
# Main Function (Hydra)
# ---------------------------
@hydra.main(config_path=".", config_name="config")
def main(cfg):
    # Initialize wandb with the config parameters
    wandb_logger = WandbLogger(project=cfg.wandb_project,
                               config=OmegaConf.to_container(cfg, resolve=True))
    
    # Instantiate the model and datamodule
    model = EDXGAN(cfg)

    transform_pipeline = transforms.Compose([
        transforms.RandomCrop(64),         # Crop the image to 64x64 pixels (center crop)
        RandomRotate(),                    # Custom random rotation transform
        transforms.RandomHorizontalFlip(), # Random horizontal flip
        transforms.ToTensor()              # Convert PIL image to a PyTorch tensor
    ])

    dm = datamodule.ArlFeAl2O3EdxDataModule(cfg.data.dir,
                                            discriminative=False, 
                                            set_aside_labels=cfg.training.set_aside_labels,
                                            batch_size=cfg.data.batch_size,
                                            num_workers=cfg.data.num_workers,
                                            repeat=cfg.data.repeat,
                                            seed=cfg.data.seed,
                                            train_transform=transform_pipeline, 
                                            val_transform=transform_pipeline, 
                                            test_transform=transform_pipeline)
    
    # Create trainer with parameters from the config and add our callback
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        devices=cfg.training.devices,
        accelerator=cfg.training.accelerator,
        logger=wandb_logger,
        callbacks=[ModelInfoCallback()]
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
