import argparse
import os
from pathlib import Path

import torch
from PIL import Image, ImageFilter

from .train_spatial_alignment import FillMaskDataset
from .trainer import get_config
from ..pipeline.flux_omini import Condition, generate
from diffusers.pipelines import FluxPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Fill-mask inference runner.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file; defaults to OMINI_CONFIG env var.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Directory containing LoRA weights (e.g., default.safetensors).",
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
        default="cuda",
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

    if dataset_config.get("type") != "fill_mask":
        raise ValueError("This inference script only supports fill_mask datasets.")

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

    adapter = args.adapter_name
    position_delta = dataset_config.get("position_delta", [0, 0])
    position_scale = dataset_config.get("position_scale", 1.0)

    def mask_path_for(relative_path: str) -> Path:
        normalized = FillMaskDataset._normalize_path(relative_path)
        parts = Path(normalized).parts
        if "images" not in parts:
            raise ValueError(
                f"Expected 'images' in path for mask replacement: {relative_path}"
            )
        replaced = ["masks" if part == "images" else part for part in parts]
        return root_dir.joinpath(*replaced).with_suffix(".png")

    with open(list_file, "r", encoding="utf-8") as handle:
        image_paths = [line.strip() for line in handle if line.strip()]

    for relative_path in image_paths:
        normalized_path = FillMaskDataset._normalize_path(relative_path)
        image_path = root_dir / normalized_path
        parts = Path(normalized_path).parts
        if "images" not in parts:
            raise ValueError(
                f"Expected 'images' in path for output replacement: {relative_path}"
            )
        output_parts = ["images_bg" if part == "images" else part for part in parts]
        output_path = root_dir.joinpath(*output_parts)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mask_path = mask_path_for(relative_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        mask = mask.point(lambda v: 255 if v > 0 else 0)
        mask = mask.filter(ImageFilter.MaxFilter(5))

        masked_image = Image.composite(
            Image.new("RGB", image.size, (0, 0, 0)),
            image,
            mask,
        )

        condition_img = masked_image.resize(condition_size)
        condition = Condition(condition_img, adapter, position_delta, position_scale)

        generator = torch.Generator(device=args.device).manual_seed(42)
        result = generate(
            pipe,
            prompt="",
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            model_config=config.get("model", {}),
            kv_cache=config.get("model", {}).get("independent_condition", False),
        )

        output_image = result.images[0].resize(image.size)
        output_image.save(output_path)


if __name__ == "__main__":
    main()
