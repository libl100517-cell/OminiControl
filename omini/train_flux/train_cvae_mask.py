import os
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from diffusers import AutoencoderKL
from PIL import Image

from .trainer import get_config
from .train_param_condition import (
    _build_param_vector,
    _estimate_mask_params,
    _normalize_path,
    _resolve_mask_path,
)


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


class CvaeMaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        list_file: str,
        root_dir: str,
        target_size=(512, 512),
        param_order=None,
        param_scale=None,
        param_categories=None,
    ):
        self.root_dir = root_dir
        self.target_size = target_size
        self.param_order = param_order or []
        self.param_scale = param_scale or {}
        self.param_categories = param_categories or {}
        self.to_tensor = T.ToTensor()
        self.image_paths = self._load_paths(list_file)

    def _load_paths(self, list_file: str) -> list[str]:
        with open(list_file, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_rel = self.image_paths[idx]
        normalized = _normalize_path(image_rel)
        mask_path = _resolve_mask_path(self.root_dir, normalized)
        mask = Image.open(mask_path).convert("L")
        mask = mask.point(lambda v: 255 if v > 0 else 0)
        mask_rgb = mask.resize(self.target_size).convert("RGB")
        params = _estimate_mask_params(mask)
        param_vector = _build_param_vector(
            params, self.param_order, self.param_scale, self.param_categories
        )
        return {
            "image": self.to_tensor(mask_rgb),
            "param_vector": torch.tensor(param_vector, dtype=torch.float32),
        }


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


def encode_mask(vae: AutoencoderKL, mask: torch.Tensor, device, dtype):
    mask = (mask * 2 - 1).to(device).to(dtype)
    latents = vae.encode(mask).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents


def decode_mask(vae: AutoencoderKL, latents: torch.Tensor, device, dtype):
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    return images.to(device).to(dtype)


def save_sample(vae, generator, p_vec, device, dtype, save_path, step):
    with torch.no_grad():
        latent = torch.randn([1, generator.latent_dim], device=device, dtype=dtype)
        z_hat = generator(latent, p_vec[:1])
        recon = decode_mask(vae, z_hat, device, dtype)
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, f"sample_step_{step}.png")
        image = (recon[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(image).save(out_path)


def main():
    config = get_config()
    training_config = config["train"]
    dataset_config = training_config["dataset"]
    device = torch.device(training_config.get("device", "cuda"))
    dtype = getattr(torch, config["dtype"])

    dataset = CvaeMaskDataset(
        list_file=dataset_config["list_file"],
        root_dir=dataset_config["root_dir"],
        target_size=dataset_config["target_size"],
        param_order=dataset_config.get("param_order", []),
        param_scale=dataset_config.get("param_scale", {}),
        param_categories=dataset_config.get("param_categories", {}),
    )

    loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 4),
        shuffle=True,
        num_workers=training_config.get("dataloader_workers", 4),
    )

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        config["flux_path"], subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)

    sample_batch = next(iter(loader))
    mask_sample = sample_batch["image"].to(device)
    latents_sample = encode_mask(vae, mask_sample, device, dtype)
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
    save_interval = training_config.get("save_interval", 1000)
    sample_interval = training_config.get("sample_interval", 1000)
    save_path = training_config.get("save_path", "runs_cvae")

    for epoch in range(epochs):
        for batch in loader:
            total_steps += 1
            masks = batch["image"].to(device)
            param_vec = batch["param_vector"].to(device)

            z_gt = encode_mask(vae, masks, device, dtype)
            z_vec = z_gt.mean(dim=(2, 3))

            p_vec = param_embed(param_vec)
            mu_q, logvar_q = posterior(z_vec, p_vec)
            mu_p, logvar_p = prior(p_vec)

            if cvae_cfg.best_of_k > 1:
                recons = []
                for _ in range(cvae_cfg.best_of_k):
                    latent = reparameterize(mu_q, logvar_q)
                    z_hat = generator(latent, p_vec)
                    recon = decode_mask(vae, z_hat, device, dtype)
                    recons.append(recon)
                recon_stack = torch.stack(recons, dim=0)
                target = masks
                recon_loss = (recon_stack - target).abs().mean(dim=(2, 3, 4))
                recon_loss = recon_loss.min(dim=0).values.mean()
            else:
                latent = reparameterize(mu_q, logvar_q)
                z_hat = generator(latent, p_vec)
                recon = decode_mask(vae, z_hat, device, dtype)
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

            if total_steps % sample_interval == 0:
                save_sample(
                    vae,
                    generator,
                    p_vec,
                    device,
                    dtype,
                    os.path.join(save_path, "samples"),
                    total_steps,
                )

            if total_steps % save_interval == 0:
                os.makedirs(save_path, exist_ok=True)
                torch.save(
                    {
                        "param_embed": param_embed.state_dict(),
                        "posterior": posterior.state_dict(),
                        "prior": prior.state_dict(),
                        "generator": generator.state_dict(),
                        "config": cvae_cfg,
                    },
                    os.path.join(save_path, f"cvae_checkpoint_{total_steps}.pt"),
                )

            if max_steps > 0 and total_steps >= max_steps:
                break
        if max_steps > 0 and total_steps >= max_steps:
            break

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
