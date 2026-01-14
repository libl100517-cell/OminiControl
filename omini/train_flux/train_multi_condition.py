import torch
import os
import random
from pathlib import Path

import torchvision.transforms as T

from PIL import Image, ImageDraw

from datasets import load_dataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, convert_to_condition, generate
from .train_spatial_alignment import FillMaskDataset, ImageConditionDataset


class ImageMultiConditionDataset(ImageConditionDataset):
    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize(self.target_size).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]

        condition_size = self.condition_size
        position_scale = self.position_scale

        condition_imgs, position_deltas = [], []
        for c_type in self.condition_type:
            condition_img, position_delta = self.__get_condition__(image, c_type)
            condition_imgs.append(condition_img.convert("RGB"))
            position_deltas.append(position_delta)

        # Randomly drop text or image (for training)
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob

        if drop_text:
            description = ""
        if drop_image:
            condition_imgs = [
                Image.new("RGB", condition_size)
                for _ in range(len(self.condition_type))
            ]

        return_dict = {
            "image": self.to_tensor(image),
            "description": description,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
        }

        for i, c_type in enumerate(self.condition_type):
            return_dict[f"condition_{i}"] = self.to_tensor(condition_imgs[i])
            return_dict[f"condition_type_{i}"] = self.condition_type[i]
            return_dict[f"position_delta_{i}"] = position_deltas[i]
            return_dict[f"position_scale_{i}"] = position_scale

        return return_dict


class FillMaskMultiConditionDataset(torch.utils.data.Dataset):
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
    ):
        self.root_dir = Path(root_dir)
        self.image_paths = self._load_paths(list_file)
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type or ["background", "mask"]
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

    def _load_paths(self, list_file: str) -> list[str]:
        with open(list_file, "r", encoding="utf-8") as handle:
            return [FillMaskDataset._normalize_path(line.strip()) for line in handle if line.strip()]

    def _mask_path(self, relative_path: str) -> Path:
        normalized_path = FillMaskDataset._normalize_path(relative_path)
        parts = Path(normalized_path).parts
        if "images" not in parts:
            raise ValueError(
                f"Expected 'images' in path for mask replacement: {relative_path}"
            )
        replaced = ["masks" if part == "images" else part for part in parts]
        return self.root_dir.joinpath(*replaced).with_suffix(".png")

    def _background_path(self, relative_path: str) -> Path:
        normalized_path = FillMaskDataset._normalize_path(relative_path)
        parts = Path(normalized_path).parts
        if "images" not in parts:
            raise ValueError(
                f"Expected 'images' in path for background replacement: {relative_path}"
            )
        replaced = ["images_bg" if part == "images" else part for part in parts]
        return self.root_dir.joinpath(*replaced)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        relative_path = self.image_paths[idx]
        image_path = self.root_dir / relative_path
        mask_path = self._mask_path(relative_path)
        background_path = self._background_path(relative_path)

        image = Image.open(image_path).convert("RGB")
        background = Image.open(background_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        if background.size != image.size:
            background = background.resize(image.size, Image.BICUBIC)
        mask = mask.point(lambda v: 255 if v > 0 else 0)

        image = image.resize(self.target_size)
        background = background.resize(self.condition_size)
        mask_rgb = mask.resize(self.condition_size).convert("RGB")

        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        description = "" if drop_text else ""

        condition_imgs = []
        for c_type in self.condition_type:
            if c_type == "mask":
                condition_imgs.append(mask_rgb)
            elif c_type in ["background", "image"]:
                condition_imgs.append(background)
            else:
                raise ValueError(f"Condition type {c_type} is not implemented.")

        if drop_image:
            condition_imgs = [
                Image.new("RGB", self.condition_size, (0, 0, 0))
                for _ in condition_imgs
            ]

        return_dict = {
            "image": self.to_tensor(image),
            "description": description,
            **({"pil_image": [image, *condition_imgs]} if self.return_pil_image else {}),
        }

        for i, c_type in enumerate(self.condition_type):
            return_dict[f"condition_{i}"] = self.to_tensor(condition_imgs[i])
            return_dict[f"condition_type_{i}"] = c_type
            return_dict[f"position_delta_{i}"] = [[0, 0]]
            return_dict[f"position_scale_{i}"] = self.position_scale

        return return_dict


@torch.no_grad()
def test_function(model, save_path, file_name):
    condition_size = model.training_config["dataset"]["condition_size"]
    target_size = model.training_config["dataset"]["target_size"]

    position_delta = model.training_config["dataset"].get("position_delta", [0, 0])
    position_scale = model.training_config["dataset"].get("position_scale", 1.0)

    condition_type = model.training_config["condition_type"]
    residual_training = model.training_config.get("residual_training", False)
    residual_alpha = model.training_config.get("residual_alpha", 1.0)
    test_list = []

    condition_list = []
    dataset_type = model.training_config["dataset"].get("type")
    if dataset_type == "fill_mask":
        list_file = model.training_config["dataset"]["list_file"]
        root_dir = Path(model.training_config["dataset"]["root_dir"])
        with open(list_file, "r", encoding="utf-8") as handle:
            relative_path = next((line.strip() for line in handle if line.strip()), None)
        if not relative_path:
            raise ValueError("fill_mask list_file is empty.")
        normalized_path = FillMaskDataset._normalize_path(relative_path)
        image_path = root_dir / normalized_path
        parts = Path(normalized_path).parts
        if "images" not in parts:
            raise ValueError(
                f"Expected 'images' in path for mask replacement: {relative_path}"
            )
        replaced = ["masks" if part == "images" else part for part in parts]
        mask_path = root_dir.joinpath(*replaced).with_suffix(".png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        mask = mask.point(lambda v: 255 if v > 0 else 0)
        background_path = root_dir.joinpath(
            *["images_bg" if part == "images" else part for part in parts]
        )
        background = Image.open(background_path).convert("RGB")
        if background.size != image.size:
            background = background.resize(image.size, Image.BICUBIC)

        for i, c_type in enumerate(condition_type):
            if c_type == "mask":
                condition_img = mask.resize(condition_size).convert("RGB")
            elif c_type in ["background", "image"]:
                condition_img = background.resize(condition_size)
            else:
                raise NotImplementedError
            condition = Condition(
                condition_img,
                model.adapter_names[i + 2],
                position_delta,
                position_scale,
            )
            condition_list.append(condition)
        test_list.append((condition_list, ""))
    else:
        for i, c_type in enumerate(condition_type):
            if c_type in ["canny", "coloring", "deblurring", "depth"]:
                image = Image.open("assets/vase_hq.jpg")
                image = image.resize(condition_size)
                condition_img = convert_to_condition(c_type, image, 5)
            elif c_type == "fill":
                condition_img = image.resize(condition_size).convert("RGB")
                w, h = image.size
                x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
                y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
                mask = Image.new("L", image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle([x1, y1, x2, y2], fill=255)
                if random.random() > 0.5:
                    mask = Image.eval(mask, lambda a: 255 - a)
                condition_img = Image.composite(
                    image, Image.new("RGB", image.size, (0, 0, 0)), mask
                )
            else:
                raise NotImplementedError
            condition = Condition(
                condition_img,
                model.adapter_names[i + 2],
                position_delta,
                position_scale,
            )
            condition_list.append(condition)
        test_list.append((condition_list, "A beautiful vase on a table."))
    os.makedirs(save_path, exist_ok=True)
    for i, (condition, prompt) in enumerate(test_list):
        generator = torch.Generator(device=model.device)
        generator.manual_seed(42)
        
        res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=condition_list,
            height=target_size[0],
            width=target_size[1],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
            residual_background=background if residual_training and dataset_type == "fill_mask" else None,
            residual_mask=mask if residual_training and dataset_type == "fill_mask" else None,
            residual_alpha=residual_alpha,
        )
        file_path = os.path.join(
            save_path, f"{file_name}_{'|'.join(condition_type)}_{i}.jpg"
        )
        res.images[0].save(file_path)


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    dataset_config = training_config["dataset"]
    if dataset_config.get("type") == "fill_mask":
        dataset = FillMaskMultiConditionDataset(
            list_file=dataset_config["list_file"],
            root_dir=dataset_config["root_dir"],
            condition_size=dataset_config["condition_size"],
            target_size=dataset_config["target_size"],
            condition_type=training_config["condition_type"],
            drop_text_prob=dataset_config["drop_text_prob"],
            drop_image_prob=dataset_config["drop_image_prob"],
            position_scale=dataset_config.get("position_scale", 1.0),
        )
    else:
        dataset = load_dataset(
            "webdataset",
            data_files={"train": dataset_config["urls"]},
            split="train",
            cache_dir="cache/t2i2m",
            num_proc=32,
        )
        dataset = ImageMultiConditionDataset(
            dataset,
            condition_size=dataset_config["condition_size"],
            target_size=dataset_config["target_size"],
            condition_type=training_config["condition_type"],
            drop_text_prob=dataset_config["drop_text_prob"],
            drop_image_prob=dataset_config["drop_image_prob"],
            position_scale=dataset_config.get("position_scale", 1.0),
        )

    cond_n = len(training_config["condition_type"])

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        adapter_names=[None, None, *["default"] * cond_n],
        # In this setting, all the conditions are using the same LoRA adapter
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
