import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import numpy as np
from pathlib import Path

from PIL import Image, ImageDraw

# from datasets import load_dataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, convert_to_condition, generate


class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size=(512, 512),
        target_size=(512, 512),
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __get_condition__(self, image, condition_type):
        condition_size = self.condition_size
        position_delta = np.array([0, 0])
        if condition_type in ["canny", "coloring", "deblurring", "depth"]:
            image, kwargs = image.resize(condition_size), {}
            if condition_type == "deblurring":
                blur_radius = random.randint(1, 10)
                kwargs["blur_radius"] = blur_radius
            condition_img = convert_to_condition(condition_type, image, **kwargs)
        elif condition_type == "depth_pred":
            depth_img = convert_to_condition("depth", image)
            condition_img = image.resize(condition_size)
            image = depth_img.resize(condition_size)
        elif condition_type == "fill":
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
        elif condition_type == "sr":
            condition_img = image.resize(condition_size)
            position_delta = np.array([0, -condition_size[0] // 16])
        else:
            raise ValueError(f"Condition type {condition_type} is not  implemented.")
        return condition_img, position_delta

    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize(self.target_size).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]

        condition_size = self.condition_size
        position_scale = self.position_scale

        condition_img, position_delta = self.__get_condition__(
            image, self.condition_type
        )

        # Randomly drop text or image (for training)
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob

        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new("RGB", condition_size, (0, 0, 0))

        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(condition_img),
            "condition_type_0": self.condition_type,
            "position_delta_0": position_delta,
            "description": description,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale_0": position_scale} if position_scale != 1.0 else {}),
        }


class FillMaskDataset(Dataset):
    def __init__(
        self,
        list_file: str,
        root_dir: str,
        condition_size=(512, 512),
        target_size=(512, 512),
        condition_type: str = "fill",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.image_paths = self._load_paths(list_file)
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def _load_paths(self, list_file: str) -> list[str]:
        with open(list_file, "r", encoding="utf-8") as handle:
            return [self._normalize_path(line.strip()) for line in handle if line.strip()]

    @staticmethod
    def _normalize_path(relative_path: str) -> str:
        return relative_path.replace("\\", "/")

    def _mask_path(self, relative_path: str) -> Path:
        normalized_path = self._normalize_path(relative_path)
        path = Path(normalized_path)
        parts = path.parts
        if "images" not in parts:
            raise ValueError(
                f"Expected 'images' in path for mask replacement: {relative_path}"
            )
        replaced = ["masks" if part == "images" else part for part in parts]
        return self.root_dir.joinpath(*replaced).with_suffix(".png")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        relative_path = self.image_paths[idx]
        normalized_path = self._normalize_path(relative_path)
        image_path = self.root_dir / normalized_path
        mask_path = self._mask_path(relative_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        mask = mask.point(lambda v: 255 if v > 0 else 0)

        masked_image = Image.composite(
            Image.new("RGB", image.size, (0, 0, 0)),
            image,
            mask,
        )

        image = image.resize(self.target_size)
        condition_img = masked_image.resize(self.condition_size)

        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        description = "" if drop_text else ""
        if drop_image:
            condition_img = Image.new("RGB", self.condition_size, (0, 0, 0))

        position_delta = np.array([0, 0])

        return {
            "image": self.to_tensor(image),
            "condition_0": self.to_tensor(condition_img),
            "condition_type_0": self.condition_type,
            "position_delta_0": position_delta,
            "description": description,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
        }


@torch.no_grad()
def test_function(model, save_path, file_name):
    condition_size = model.training_config["dataset"]["condition_size"]
    target_size = model.training_config["dataset"]["target_size"]

    position_delta = model.training_config["dataset"].get("position_delta", [0, 0])
    position_scale = model.training_config["dataset"].get("position_scale", 1.0)

    adapter = model.adapter_names[2]
    condition_type = model.training_config["condition_type"]
    test_list = []

    if condition_type in ["canny", "coloring", "deblurring", "depth"]:
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition_img = convert_to_condition(condition_type, image, 5)
        condition = Condition(condition_img, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "depth_pred":
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition = Condition(image, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    elif condition_type == "fill":
        dataset_type = model.training_config["dataset"].get("type")
        if dataset_type == "fill_mask":
            list_file = model.training_config["dataset"]["list_file"]
            root_dir = Path(model.training_config["dataset"]["root_dir"])
            with open(list_file, "r", encoding="utf-8") as handle:
                relative_path = next(
                    (line.strip() for line in handle if line.strip()), None
                )
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
            mask_path = root_dir.joinpath(*replaced)

            image = Image.open(image_path).convert("RGB").resize(condition_size)
            mask = Image.open(mask_path).convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
            mask = mask.point(lambda v: 255 if v > 0 else 0)
            condition_img = Image.composite(
                Image.new("RGB", image.size, (0, 0, 0)), image, mask
            )
        else:
            condition_img = (
                Image.open("./assets/vase_hq.jpg").resize(condition_size).convert("RGB")
            )
            mask = Image.new("L", condition_img.size, 0)
            draw = ImageDraw.Draw(mask)
            a = condition_img.size[0] // 4
            b = a * 3
            draw.rectangle([a, a, b, b], fill=255)
            condition_img = Image.composite(
                condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
            )
        condition = Condition(condition_img, adapter, position_delta, position_scale)
        prompt = "" if dataset_type == "fill_mask" else "A beautiful vase on a table."
        test_list.append((condition, prompt))
    elif condition_type == "super_resolution":
        image = Image.open("assets/vase_hq.jpg")
        image = image.resize(condition_size)
        condition = Condition(image, adapter, position_delta, position_scale)
        test_list.append((condition, "A beautiful vase on a table."))
    else:
        raise NotImplementedError
    os.makedirs(save_path, exist_ok=True)
    for i, (condition, prompt) in enumerate(test_list):
        generator = torch.Generator(device=model.device)
        generator.manual_seed(42)

        res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
        )
        file_path = os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
        res.images[0].save(file_path)


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    dataset_config = training_config["dataset"]
    if dataset_config.get("type") == "fill_mask":
        dataset = FillMaskDataset(
            list_file=dataset_config["list_file"],
            root_dir=dataset_config["root_dir"],
            condition_size=dataset_config["condition_size"],
            target_size=dataset_config["target_size"],
            condition_type=training_config["condition_type"],
            drop_text_prob=dataset_config["drop_text_prob"],
            drop_image_prob=dataset_config["drop_image_prob"],
        )
    # else:
    #     # Load dataset text-to-image-2M
    #     dataset = load_dataset(
    #         "webdataset",
    #         data_files={"train": dataset_config["urls"]},
    #         split="train",
    #         cache_dir="cache/t2i2m",
    #         num_proc=32,
    #     )

        # # Initialize custom dataset
        # dataset = ImageConditionDataset(
        #     dataset,
        #     condition_size=dataset_config["condition_size"],
        #     target_size=dataset_config["target_size"],
        #     condition_type=training_config["condition_type"],
        #     drop_text_prob=dataset_config["drop_text_prob"],
        #     drop_image_prob=dataset_config["drop_image_prob"],
        #     position_scale=dataset_config.get("position_scale", 1.0),
        # )

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    train(dataset, trainable_model, config, test_function)


if __name__ == "__main__":
    main()
