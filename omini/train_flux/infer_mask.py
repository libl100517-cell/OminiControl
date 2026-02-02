import argparse
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

from diffusers.pipelines import FluxPipeline

from .trainer import get_config
from ..pipeline.flux_omini import Condition, generate
from .train_param_condition import (
    _build_param_vector,
    _estimate_mask_params,
    _normalize_path,
    _resolve_background_path,
    _resolve_mask_path,
)


def _resolve_text_embed_dim(transformer) -> int:
    for attr in ("cross_attention_dim", "joint_attention_dim", "hidden_size"):
        value = getattr(transformer.config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Unable to resolve text embedding dimension for param_condition.")


def parse_args():
    parser = argparse.ArgumentParser(description="Mask inference with param conditioning.")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/libaoluo/sam2/OminiControl/train/config/param_condition.yaml",
        help="Path to config file; defaults to OMINI_CONFIG env var.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/home/libaoluo/sam2/OminiControl/runs/20260114-161713/ckpt/2000",
        help="Directory containing LoRA weights (e.g., default.safetensors).",
    )
    parser.add_argument(
        "--param_mlp_path",
        type=str,
        default="",
        help="Path to param_mlp.pt; defaults to <lora_path>/param_mlp.pt.",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="default",
        help="LoRA adapter name to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="infer_vis_mask",
        help="Directory to save mosaic outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def mask_to_rgb_viz(mask_l: Image.Image) -> Image.Image:
    mask_l = mask_l.convert("L")
    return Image.merge("RGB", (mask_l, mask_l, mask_l))


def overlay_red(base_rgb: Image.Image, mask_l: Image.Image, alpha: float = 0.45) -> Image.Image:
    base = base_rgb.convert("RGBA")
    mask_l = mask_l.convert("L")
    a = mask_l.point(lambda v: int(v * alpha))
    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red.putalpha(a)
    out = Image.alpha_composite(base, red)
    return out.convert("RGB")


def make_2x2_mosaic(im00, im01, im10, im11) -> Image.Image:
    w, h = im00.size
    canvas = Image.new("RGB", (w * 2, h * 2))
    canvas.paste(im00, (0, 0))
    canvas.paste(im01, (w, 0))
    canvas.paste(im10, (0, h))
    canvas.paste(im11, (w, h))
    return canvas


def main():
    args = parse_args()
    if args.config:
        os.environ["OMINI_CONFIG"] = args.config

    config = get_config()
    training_config = config["train"]
    dataset_config = training_config["dataset"]

    list_file = dataset_config["list_file"]
    root_dir = Path(dataset_config["root_dir"])
    condition_size = tuple(dataset_config["condition_size"])
    target_size = tuple(dataset_config["target_size"])

    pipe: FluxPipeline = FluxPipeline.from_pretrained(
        config["flux_path"], torch_dtype=getattr(torch, config["dtype"])
    ).to(args.device)

    pipe.load_lora_weights(
        args.lora_path,
        weight_name=f"{args.adapter_name}.safetensors",
        adapter_name=args.adapter_name,
    )
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters([args.adapter_name])

    param_condition = config.get("model", {}).get("param_condition", {})
    param_mlp = None
    if param_condition.get("enabled", False):
        input_dim = int(param_condition.get("vector_dim", 0))
        hidden_dim = int(param_condition.get("hidden_dim", 128))
        embed_dim = param_condition.get("embed_dim")
        if embed_dim is None:
            embed_dim = _resolve_text_embed_dim(pipe.transformer)
        param_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, embed_dim),
        ).to(args.device)
        param_mlp_path = args.param_mlp_path or os.path.join(args.lora_path, "param_mlp.pt")
        if os.path.exists(param_mlp_path):
            param_mlp.load_state_dict(torch.load(param_mlp_path, map_location=args.device))
        param_mlp.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(list_file, "r", encoding="utf-8") as handle:
        image_paths = [line.strip() for line in handle if line.strip()]

    for idx, relative_path in enumerate(image_paths):
        normalized_path = _normalize_path(relative_path)
        image_path = root_dir / normalized_path
        mask_path = _resolve_mask_path(root_dir, normalized_path)
        background_path = _resolve_background_path(root_dir, normalized_path, "")

        background = ImageOps.exif_transpose(Image.open(background_path)).convert("RGB")
        mask = ImageOps.exif_transpose(Image.open(mask_path)).convert("L")
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        if mask.size != background.size:
            mask = mask.resize(background.size, Image.NEAREST)
        if image.size != background.size:
            image = image.resize(background.size, Image.BICUBIC)

        params = _estimate_mask_params(mask)
        param_vector = _build_param_vector(
            params,
            dataset_config.get("param_order", []),
            dataset_config.get("param_scale", {}),
            dataset_config.get("param_categories", {}),
        )

        background_cond = background.resize(condition_size)
        condition = Condition(background_cond, args.adapter_name, [0, 0], 1.0)

        extra_prompt_embeds = None
        if param_mlp is not None:
            vector_tensor = torch.tensor([param_vector], device=args.device, dtype=pipe.dtype)
            extra_prompt_embeds = param_mlp(vector_tensor)

        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        description = ""
        result = generate(
            pipe,
            prompt=description,
            conditions=[condition],
            height=target_size[0],
            width=target_size[1],
            generator=generator,
            model_config=config.get("model", {}),
            kv_cache=config.get("model", {}).get("independent_condition", False),
            extra_prompt_embeds=extra_prompt_embeds,
        )

        output_image = result.images[0].resize(background.size)

        bg_tile = background.copy()
        mask_tile = mask_to_rgb_viz(mask)
        gen_tile = output_image
        overlay_tile = overlay_red(output_image, mask, alpha=0.45)
        mosaic = make_2x2_mosaic(bg_tile, mask_tile, gen_tile, overlay_tile)

        safe_name = normalized_path.replace("/", "__").replace("\\", "__")
        out_path = out_dir / f"{idx:05d}__{safe_name}.jpg"
        mosaic.save(out_path, quality=95)

    print(f"Done. Saved {len(image_paths)} mosaics to: {out_dir}")


if __name__ == "__main__":
    main()
