import os
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from diffusers.pipelines import FluxPipeline

from .trainer import get_config
from .train_param_condition import ParamConditionDataset


@dataclass
class CvaeConfig:
    latent_channels: int
    latent_height: int
    latent_width: int
    param_dim: int
    param_embed_dim: int
    latent_dim: int
    best_of_k: int
    kl_weight: float
    kl_warmup_steps: int


class PosteriorNet(nn.Module):
    def __init__(self, z_dim: int, p_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + p_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.to_mu = nn.Linear(256, latent_dim)
        self.to_logvar = nn.Linear(256, latent_dim)

    def forward(self, z_vec: torch.Tensor, p_vec: torch.Tensor):
        h = self.net(torch.cat([z_vec, p_vec], dim=-1))
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-8.0, 4.0)
        return mu, logvar


class PriorNet(nn.Module):
    def __init__(self, p_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(p_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.to_mu = nn.Linear(256, latent_dim)
        self.to_logvar = nn.Linear(256, latent_dim)

    def forward(self, p_vec: torch.Tensor):
        h = self.net(p_vec)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-8.0, 4.0)
        return mu, logvar


class GeneratorNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        p_dim: int,
        latent_channels: int,
        latent_height: int,
        latent_width: int,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.base = nn.Sequential(
            nn.Linear(latent_dim, latent_channels * latent_height * latent_width),
            nn.SiLU(),
        )
        self.film = nn.Sequential(
            nn.Linear(p_dim, 256),
            nn.SiLU(),
            nn.Linear(256, latent_channels * 2),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1),
        )

    def forward(self, latent: torch.Tensor, p_vec: torch.Tensor):
        base = self.base(latent).view(
            latent.shape[0],
            self.latent_channels,
            self.latent_height,
            self.latent_width,
        )
        gamma_beta = self.film(p_vec)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(latent.shape[0], self.latent_channels, 1, 1)
        beta = beta.view(latent.shape[0], self.latent_channels, 1, 1)
        z_hat = gamma * base + beta
        return z_hat + self.refine(z_hat)


class ParamEmbed(nn.Module):
    def __init__(self, param_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.net(params)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * eps


def kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
    return 0.5 * (
        logvar_p
        - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p)
        - 1.0
    ).sum(dim=-1)


def encode_mask(pipe: FluxPipeline, mask: torch.Tensor, device, dtype):
    mask = pipe.image_processor.preprocess(mask).to(device).to(dtype)
    latents = pipe.vae.encode(mask).latent_dist.sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return latents


def decode_mask(pipe: FluxPipeline, latents: torch.Tensor, device, dtype):
    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    images = pipe.vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    return images.to(device).to(dtype)


def main():
    config = get_config()
    training_config = config["train"]
    dataset_config = training_config["dataset"]
    device = torch.device(training_config.get("device", "cuda"))
    dtype = getattr(torch, config["dtype"])

    dataset = ParamConditionDataset(
        list_file=dataset_config["list_file"],
        root_dir=dataset_config["root_dir"],
        condition_size=dataset_config["condition_size"],
        target_size=dataset_config["target_size"],
        condition_type=training_config["condition_type"],
        drop_text_prob=dataset_config["drop_text_prob"],
        drop_image_prob=dataset_config["drop_image_prob"],
        position_scale=dataset_config.get("position_scale", 1.0),
        param_order=dataset_config.get("param_order", []),
        param_scale=dataset_config.get("param_scale", {}),
        param_categories=dataset_config.get("param_categories", {}),
        prompt=dataset_config.get("prompt", ""),
    )

    loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 4),
        shuffle=True,
        num_workers=training_config.get("dataloader_workers", 4),
    )

    pipe: FluxPipeline = FluxPipeline.from_pretrained(
        config["flux_path"], torch_dtype=dtype
    ).to(device)
    pipe.vae.eval()
    pipe.vae.requires_grad_(False)

    sample_batch = next(iter(loader))
    mask_sample = sample_batch["image"].to(device)
    latents_sample = encode_mask(pipe, mask_sample, device, dtype)
    latent_channels = latents_sample.shape[1]
    latent_height = latents_sample.shape[2]
    latent_width = latents_sample.shape[3]

    model_config = config.get("model", {}).get("cvae", {})
    cvae_cfg = CvaeConfig(
        latent_channels=latent_channels,
        latent_height=latent_height,
        latent_width=latent_width,
        param_dim=model_config.get("param_dim", sample_batch["param_vector"].shape[1]),
        param_embed_dim=model_config.get("param_embed_dim", 128),
        latent_dim=model_config.get("latent_dim", 64),
        best_of_k=model_config.get("best_of_k", 1),
        kl_weight=model_config.get("kl_weight", 1.0),
        kl_warmup_steps=model_config.get("kl_warmup_steps", 1000),
    )

    param_embed = ParamEmbed(cvae_cfg.param_dim, cvae_cfg.param_embed_dim).to(device)
    posterior = PosteriorNet(
        latent_channels, cvae_cfg.param_embed_dim, cvae_cfg.latent_dim
    ).to(device)
    prior = PriorNet(cvae_cfg.param_embed_dim, cvae_cfg.latent_dim).to(device)
    generator = GeneratorNet(
        cvae_cfg.latent_dim,
        cvae_cfg.param_embed_dim,
        cvae_cfg.latent_channels,
        cvae_cfg.latent_height,
        cvae_cfg.latent_width,
    ).to(device)

    params = list(param_embed.parameters()) + list(posterior.parameters())
    params += list(prior.parameters()) + list(generator.parameters())
    optimizer = torch.optim.AdamW(params, lr=training_config.get("lr", 1e-4))

    total_steps = 0
    max_steps = training_config.get("max_steps", -1)
    epochs = training_config.get("epochs", 1)

    for epoch in range(epochs):
        for batch in loader:
            total_steps += 1
            masks = batch["image"].to(device)
            param_vec = batch["param_vector"].to(device)

            z_gt = encode_mask(pipe, masks, device, dtype)
            z_vec = z_gt.mean(dim=(2, 3))

            p_vec = param_embed(param_vec)
            mu_q, logvar_q = posterior(z_vec, p_vec)
            mu_p, logvar_p = prior(p_vec)

            if cvae_cfg.best_of_k > 1:
                recons = []
                for _ in range(cvae_cfg.best_of_k):
                    latent = reparameterize(mu_q, logvar_q)
                    z_hat = generator(latent, p_vec)
                    recon = decode_mask(pipe, z_hat, device, dtype)
                    recons.append(recon)
                recon_stack = torch.stack(recons, dim=0)
                target = masks
                recon_loss = (recon_stack - target).abs().mean(dim=(2, 3, 4))
                recon_loss = recon_loss.min(dim=0).values.mean()
            else:
                latent = reparameterize(mu_q, logvar_q)
                z_hat = generator(latent, p_vec)
                recon = decode_mask(pipe, z_hat, device, dtype)
                recon_loss = (recon - masks).abs().mean()

            kl = kl_divergence(mu_q, logvar_q, mu_p, logvar_p).mean()
            kl_weight = (
                cvae_cfg.kl_weight
                * min(1.0, total_steps / max(1, cvae_cfg.kl_warmup_steps))
            )
            loss = recon_loss + kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_steps % training_config.get("log_interval", 10) == 0:
                print(
                    f"Epoch {epoch} Step {total_steps} "
                    f"Loss {loss.item():.4f} Recon {recon_loss.item():.4f} "
                    f"KL {kl.item():.4f} (w={kl_weight:.3f})"
                )

            if max_steps > 0 and total_steps >= max_steps:
                break
        if max_steps > 0 and total_steps >= max_steps:
            break

    save_path = training_config.get("save_path", "runs_cvae")
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        {
            "param_embed": param_embed.state_dict(),
            "posterior": posterior.state_dict(),
            "prior": prior.state_dict(),
            "generator": generator.state_dict(),
            "config": cvae_cfg,
        },
        os.path.join(save_path, "cvae_checkpoint.pt"),
    )


if __name__ == "__main__":
    main()
