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
    return parser.parse_args()


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

    with open(list_file, "r", encoding="utf-8") as handle:
        image_paths = [line.strip() for line in handle if line.strip()]

    for relative_path in image_paths:
        normalized_path = _normalize_path(relative_path)
        image_path = root_dir / normalized_path
        mask_path = _resolve_mask_path(root_dir, normalized_path)
        background_path = _resolve_background_path(root_dir, normalized_path, "")

        background = ImageOps.exif_transpose(Image.open(background_path)).convert("RGB")
        mask = ImageOps.exif_transpose(Image.open(mask_path)).convert("L")
        if mask.size != background.size:
            mask = mask.resize(background.size, Image.NEAREST)

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

        generator = torch.Generator(device=args.device).manual_seed(42)
        result = generate(
            pipe,
            prompt=dataset_config.get("prompt", ""),
            conditions=[condition],
            height=target_size[0],
            width=target_size[1],
            generator=generator,
            model_config=config.get("model", {}),
            kv_cache=config.get("model", {}).get("independent_condition", False),
            extra_prompt_embeds=extra_prompt_embeds,
        )

        output_parts = [
            "images_mask_pred" if part == "images" else part
            for part in Path(normalized_path).parts
        ]
        output_path = root_dir.joinpath(*output_parts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_image = result.images[0].resize(background.size)
        output_image.save(output_path)


if __name__ == "__main__":
    main()
