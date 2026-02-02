import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import cv2

from PIL import Image

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, generate


def _normalize_path(p: str) -> str:
    return p.replace("\\", "/").lstrip("/")


def _parse_list_line(line: str) -> tuple[str, str]:
    parts = [part.strip() for part in line.split("\t") if part.strip()]
    if len(parts) == 1:
        parts = [part.strip() for part in line.split(",") if part.strip()]
    if len(parts) < 1:
        raise ValueError("Each list entry must include image_path.")
    image_path = parts[0]
    background_path = parts[1] if len(parts) > 1 else ""
    return image_path, background_path


def _resolve_background_path(root_dir: Path, image_path: str, background_path: str) -> Path:
    if background_path:
        return root_dir / _normalize_path(background_path)
    normalized = _normalize_path(image_path)
    parts = Path(normalized).parts
    if "images" not in parts:
        raise ValueError(f"Expected 'images' in path for background replacement: {image_path}")
    replaced = ["images_bg" if part == "images" else part for part in parts]
    return root_dir.joinpath(*replaced)


def _resolve_mask_path(root_dir: Path, image_path: str) -> Path:
    normalized = _normalize_path(image_path)
    parts = Path(normalized).parts
    if "images" not in parts:
        raise ValueError(f"Expected 'images' in path for mask replacement: {image_path}")
    replaced = ["masks" if part == "images" else part for part in parts]
    return root_dir.joinpath(*replaced).with_suffix(".png")


def _build_param_vector(
    params: dict,
    param_order: list[str],
    param_scale: dict,
    param_categories: dict,
) -> list[float]:
    values = []
    for name in param_order:
        raw = params.get(name, 0)
        if isinstance(raw, str):
            categories = param_categories.get(name, [])
            if categories and raw in categories:
                value = categories.index(raw)
            else:
                value = 0.0
        else:
            value = float(raw)
        scale = float(param_scale.get(name, 1.0))
        if scale != 0:
            value = value / scale
        values.append(value)
    return values


def _distance_map(mask: Image.Image) -> Image.Image:
    mask_np = np.array(mask, dtype=np.uint8)
    mask_bin = (mask_np > 0).astype(np.uint8) * 255
    inv = 255 - mask_bin
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    max_val = dist.max() if dist.max() > 0 else 1.0
    dist = dist / max_val
    dist_img = (dist * 255).astype(np.uint8)
    return Image.fromarray(dist_img, mode="L")


def _mask_to_binary(mask: Image.Image) -> np.ndarray:
    mask_np = np.array(mask, dtype=np.uint8)
    return (mask_np > 0).astype(np.uint8)


def _estimate_mask_params(mask: Image.Image) -> dict:
    mask_bin = _mask_to_binary(mask)
    if mask_bin.sum() == 0:
        return {
            "length": 0.0,
            "avg_width": 0.0,
            "max_width": 0.0,
            "count": 0,
            "shape": "mesh-like",
            "branch": 0,
        }

    dist = cv2.distanceTransform(255 - mask_bin * 255, cv2.DIST_L2, 3)
    widths = dist[mask_bin.astype(bool)] * 2.0
    avg_width = float(np.mean(widths)) if widths.size else 0.0
    max_width = float(np.max(widths)) if widths.size else 0.0

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = len(contours)
    length = float(sum(cv2.arcLength(cnt, False) for cnt in contours))

    points = np.column_stack(np.where(mask_bin > 0))
    shape = "mesh-like"
    if points.shape[0] >= 2:
        centered = points.astype(np.float32)
        centered -= centered.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        ratio = (eigvals[1] + 1e-6) / (eigvals[0] + 1e-6)
        if ratio > 3.0 and count <= 1:
            main_vec = eigvecs[:, np.argmax(eigvals)]
            vertical = np.array([0.0, 1.0], dtype=np.float32)
            main_norm = np.linalg.norm(main_vec) + 1e-6
            cos_angle = float(np.clip(np.dot(main_vec, vertical) / main_norm, -1.0, 1.0))
            angle = np.degrees(np.arccos(abs(cos_angle)))
            if angle < 22.5 or angle > 157.5:
                shape = "Horizontal"
            elif 67.5 < angle < 112.5:
                shape = "vertical"
            else:
                shape = "diagonal"

    branch = max(0, count - 1)
    if count > 2:
        shape = "mesh-like"

    return {
        "length": length,
        "avg_width": avg_width,
        "max_width": max_width,
        "count": count,
        "shape": shape,
        "branch": branch,
    }


class ParamConditionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        list_file: str,
        root_dir: str,
        condition_size=(512, 512),
        target_size=(512, 512),
        condition_type=None,
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
        param_order: list[str] | None = None,
        param_scale: dict | None = None,
        param_categories: dict | None = None,
        prompt: str = "",
    ):
        self.root_dir = Path(root_dir)
        self.entries = self._load_entries(list_file)
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type or ["background"]
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale
        self.param_order = param_order or []
        self.param_scale = param_scale or {}
        self.param_categories = param_categories or {}
        self.prompt = prompt
        self.to_tensor = T.ToTensor()

    def _load_entries(self, list_file: str) -> list[tuple[str, str]]:
        with open(list_file, "r", encoding="utf-8") as handle:
            entries = []
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entries.append(_parse_list_line(line))
            return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image_rel, background_rel = self.entries[idx]
        image_path = self.root_dir / _normalize_path(image_rel)
        mask_path = _resolve_mask_path(self.root_dir, image_rel)
        background_path = _resolve_background_path(self.root_dir, image_rel, background_rel)

        image = Image.open(image_path).convert("RGB")
        background = Image.open(background_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        if background.size != image.size:
            background = background.resize(image.size, Image.BICUBIC)

        params = _estimate_mask_params(mask)
        param_vector = _build_param_vector(
            params, self.param_order, self.param_scale, self.param_categories
        )
        mask_rgb = mask.resize(self.target_size).convert("RGB")
        background = background.resize(self.condition_size)

        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        description = "" if drop_text else self.prompt

        condition_imgs = []
        for c_type in self.condition_type:
            if c_type in ["background", "image"]:
                condition_imgs.append(background)
            else:
                raise ValueError(f"Condition type {c_type} is not implemented.")

        if drop_image:
            condition_imgs = [
                Image.new("RGB", self.condition_size, (0, 0, 0))
                for _ in condition_imgs
            ]

        return_dict = {
            "image": self.to_tensor(mask_rgb),
            "description": description,
            "param_vector": torch.tensor(param_vector, dtype=torch.float32),
            **({"pil_image": [image, *condition_imgs]} if self.return_pil_image else {}),
        }

        for i, c_type in enumerate(self.condition_type):
            return_dict[f"condition_{i}"] = self.to_tensor(condition_imgs[i])
            return_dict[f"condition_type_{i}"] = c_type
            return_dict[f"position_delta_{i}"] = np.array([0, 0])
            return_dict[f"position_scale_{i}"] = self.position_scale

        return return_dict


@torch.no_grad()
def test_function(model, save_path, file_name):
    condition_size = model.training_config["dataset"]["condition_size"]
    target_size = model.training_config["dataset"]["target_size"]
    dataset_config = model.training_config["dataset"]
    condition_type = model.training_config["condition_type"]

    list_file = dataset_config["list_file"]
    root_dir = Path(dataset_config["root_dir"])
    with open(list_file, "r", encoding="utf-8") as handle:
        first_line = next((line.strip() for line in handle if line.strip()), None)
    if not first_line:
        raise ValueError("param_condition list_file is empty.")

    image_rel, background_rel = _parse_list_line(first_line)
    image_path = root_dir / _normalize_path(image_rel)
    mask_path = _resolve_mask_path(root_dir, image_rel)
    background_path = _resolve_background_path(root_dir, image_rel, background_rel)
    image = Image.open(image_path).convert("RGB")
    background = Image.open(background_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    if background.size != image.size:
        background = background.resize(image.size, Image.BICUBIC)

    params = _estimate_mask_params(mask)
    param_vector = _build_param_vector(
        params,
        dataset_config.get("param_order", []),
        dataset_config.get("param_scale", {}),
        dataset_config.get("param_categories", {}),
    )

    condition_list = []
    for i, c_type in enumerate(condition_type):
        if c_type in ["background", "image"]:
            condition_img = background.resize(condition_size)
        else:
            raise NotImplementedError
        condition = Condition(
            condition_img,
            model.adapter_names[i + 2],
            [0, 0],
            dataset_config.get("position_scale", 1.0),
        )
        condition_list.append(condition)

    extra_prompt_embeds = None
    if model.param_mlp is not None:
        vector_tensor = torch.tensor([param_vector], device=model.device, dtype=model.dtype)
        extra_prompt_embeds = model.param_mlp(vector_tensor)

    os.makedirs(save_path, exist_ok=True)
    generator = torch.Generator(device=model.device).manual_seed(42)
    res = generate(
        model.flux_pipe,
        prompt=dataset_config.get("prompt", ""),
        conditions=condition_list,
        height=target_size[0],
        width=target_size[1],
        generator=generator,
        model_config=model.model_config,
        kv_cache=model.model_config.get("independent_condition", False),
        extra_prompt_embeds=extra_prompt_embeds,
    )
    res = res.images[0].resize(image.size)
    res.save(os.path.join(save_path, f"{file_name}_param_condition.jpg"))


def main():
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    dataset_config = training_config["dataset"]
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

    cond_n = len(training_config["condition_type"])

    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device="cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        adapter_names=[None, None, *["default"] * cond_n],
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
