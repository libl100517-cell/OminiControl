import torch
import os
import random
from pathlib import Path
import numpy as np

import torchvision.transforms as T

from PIL import Image, ImageDraw

# from datasets import load_dataset

from .trainer import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, convert_to_condition, generate
from .train_spatial_alignment import FillMaskDataset, ImageConditionDataset


# ---------------------------
# Dataset -> (scene, material, viewpoint)
# viewpoint: ground_closeup / uav_aerial / unknown
# scene: road / bridge / building / tunnel / dam / tile / industrial / infrastructure / mixed / none
# material: asphalt / concrete / steel / stone / masonry / ceramic / mixed / unknown
# ---------------------------

UAV_DATASETS = {"UAV102", "UAV315", "UAV5k", "UAV75"}

# 你的表格 + 我帮你归一后的映射（可继续补充）
DATASET_META = {
    # Asphalt road
    "Asphalt3k":         ("road", "asphalt", "vehicle_inspection", "ccd_camera"),
    "AsphaltCrack300":   ("road", "asphalt", "ground_closeup", "handheld_camera"),
    "BeijingHighway465": ("road", "asphalt", "vehicle_inspection", "vehicle_camera"),
    "CFD118":            ("road", "asphalt", "ground_closeup", "mobile_phone"),
    "CRACK500":          ("road", "asphalt", "ground_closeup", "mobile_phone"),
    "CrackLS315":        ("road", "asphalt", "line_scan_active_light", "line_scan_camera"),
    "CrackMap120":       ("road", "asphalt", "vehicle_inspection", "area_scan_camera"),
    "CrackSC197":        ("road", "asphalt", "ground_closeup", "mobile_phone"),
    "CrackSeg3k":        ("road", "asphalt", "vehicle_inspection", "vehicle_camera"),
    "CrackTree260":      ("road", "asphalt", "visible_light", "area_scan_camera"),
    "CrackTree206":      ("road", "asphalt", "visible_light", "area_scan_camera"),
    "CRKWH100":          ("road", "asphalt", "visible_light", "line_scan_camera"),
    "EdmCrack600":       ("road", "asphalt", "vehicle_inspection", "area_scan_camera"),
    "GAPS384":           ("road", "asphalt", "vehicle_inspection", "vehicle_camera"),
    "ShadowCrack210":    ("road", "asphalt", "ground_closeup", "mobile_phone"),
    "SUT130":            ("road", "asphalt", "vehicle_inspection", "vehicle_camera"),
    "AigleRN":           ("road", "asphalt", "ground_closeup", "handheld_camera"),
    "ESAR":              ("road", "asphalt", "ground_closeup", "handheld_camera"),
    "LCMS":              ("road", "asphalt", "ground_closeup", "handheld_camera"),
    "AEL58":             ("road", "asphalt", "vehicle_inspection", "vehicle_camera"),
    "GaMM37":            ("road", "asphalt", "vehicle_inspection", "vehicle_camera"),
    "Sylvie":            ("road", "asphalt", "ground_closeup", "handheld_camera"),

    # Concrete road
    "CCSS670":           ("road", "concrete", "ground_closeup", "ccd_camera"),
    "CDLN1k":            ("road", "concrete", "unknown_source", "handheld_camera"),
    "NHA12D80":          ("road", "concrete", "vehicle_inspection", "ccd_camera"),
    "Road420":           ("road", "concrete", "ground_closeup", "mobile_phone"),
    "LSCD143":           ("road", "mixed", "ground_closeup", "mobile_phone"),

    # Bridge
    "BCL11k":            ("bridge", "unknown", "ground_closeup", "handheld_camera"),  # material by filename prefix
    "CSB1k":             ("bridge", "steel", "unknown_source", "handheld_camera"),
    "S2DS743":           ("bridge", "concrete", "uav_aerial", "uav_camera"),
    "LCW4k":             ("bridge", "mixed", "mixed_source", "internet_image"),
    "NCCD5k":            ("infrastructure", "concrete", "ground_closeup", "handheld_camera"),

    # Building / concrete
    "BuildCrack358":     ("building", "concrete", "uav_aerial", "uav_camera"),
    "Facade390":         ("building", "concrete", "uav_aerial", "uav_camera"),
    "FCNCrack776":       ("building", "concrete", "mixed_source", "internet_image"),
    "Concrete3k":        ("building", "concrete", "ground_closeup", "handheld_camera"),
    "Concrete600":       ("building", "concrete", "ground_closeup", "handheld_camera"),
    "KaggleConcrete458": ("building", "concrete", "ground_closeup", "handheld_camera"),

    # Tunnel / concrete
    "CrackTAV484":       ("tunnel", "concrete", "unknown_source", "handheld_camera"),
    "KICT200":           ("tunnel", "concrete", "vehicle_inspection", "ccd_camera"),
    "Tunnel200":         ("tunnel", "concrete", "ground_closeup", "mobile_phone"),
    "TunnelCrack919":    ("tunnel", "concrete", "ground_closeup", "ccd_camera"),

    # Dam / concrete
    "DSI2k":             ("dam", "concrete", "ground_closeup", "industrial_camera"),

    # Stone / masonry / ceramic
    "DIC530":            ("building", "masonry", "ground_closeup", "dic_camera"),
    "Masonry240":        ("building", "masonry", "mixed_source", "mobile_phone"),
    "MCS246":            ("building", "stone", "ground_closeup", "mobile_phone"),
    "Stone331":          ("road", "stone", "visible_light", "area_scan_camera"),
    "Ceramic100":        ("tile", "ceramic", "unknown_source", "handheld_camera"),
    "TopoDS7k":          ("building", "mixed", "unknown_source", "handheld_camera"),
    "CrSpEE2k":          ("building", "mixed", "ground_closeup", "handheld_camera"),
    "CSSC186":           ("building", "mixed", "unknown_source", "internet_image"),
    "StructureCrack690": ("building", "mixed", "mixed_source", "internet_image"),

    # Mixed / infrastructure
    "CrackNJ156":        ("road", "mixed", "ground_closeup", "ccd_camera"),
    "DeepCrack537":      ("infrastructure", "mixed", "unknown_source", "internet_image"),
    "Kaggle800":         ("infrastructure", "mixed", "ground_closeup", "handheld_camera"),
    "TUT1k":             ("mixed", "mixed", "mixed_source", "mobile_phone"),
    "SCCD7k":            ("mixed", "unknown", "mixed_source", "internet_image"),
    "CrackVision12K":    ("mixed", "unknown", "unknown_source", "handheld_camera"),
    "CrackSeg9k":        ("mixed", "unknown", "unknown_source", "handheld_camera"),
    "OmniCrack30k":      ("mixed", "unknown", "unknown_source", "handheld_camera"),
    "UDTIRICrack2k":     ("mixed", "unknown", "unknown_source", "handheld_camera"),
    "Khanh11k":          ("mixed", "unknown", "unknown_source", "handheld_camera"),
    "Conglomerate11k":   ("mixed", "unknown", "unknown_source", "handheld_camera"),
    "CCrack3k":          ("mixed", "unknown", "unknown_source", "handheld_camera"),
    "SteeCrack4k":       ("mixed", "unknown", "ground_closeup", "handheld_camera"),
    "SteelCrack50":      ("mixed", "unknown", "ground_closeup", "handheld_camera"),

    # UAV
    "UAV102":            ("road", "asphalt", "uav_aerial", "uav_camera"),
    "UAV5k":             ("road", "asphalt", "uav_aerial", "uav_camera"),
    "UAV315":            ("dam", "concrete", "uav_aerial", "uav_camera"),
    "UAV75":             ("bridge", "concrete", "uav_aerial", "uav_camera"),
}

def _normalize_path(p: str) -> str:
    # 兼容 Windows 反斜杠
    return p.replace("\\", "/").lstrip("/")

def build_prompt(relative_path: str) -> str:
    """
    Example:
      Asphalt3k\\images\\crack_237.jpg
      BCL11k\\images\\c_000123.jpg
      BCL11k\\images\\s_000123.jpg
    """
    rp = _normalize_path(relative_path)
    parts = rp.split("/")
    dataset = parts[0] if parts else "Unknown"
    fname = Path(parts[-1]).name if parts else ""

    scene, material, mode, device = DATASET_META.get(dataset, ("mixed", "unknown", "unknown_source", "handheld_camera"))

    # UAV override
    if dataset in UAV_DATASETS:
        mode = "uav_aerial"
        device = "uav_camera"

    # BCL11k: filename prefix decides material
    if dataset == "BCL11k":
        c0 = (fname[:1] or "").lower()
        if c0 == "c":
            material = "concrete"
        elif c0 == "s":
            material = "steel"
        else:
            material = "unknown"

    # 你说“别太长”，所以只加最关键的 domain token：dataset + mode + device
    dataset_tag = f"[DATASET={dataset}]"

    if scene == "none":
        return f"{dataset_tag} {mode} {device} intact no_crack"

    return f"{dataset_tag} {mode} {device} {scene} {material} crack_texture"

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
        # description = "" if drop_text else ""
        description = build_prompt(relative_path) if not drop_text else ""

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
            return_dict[f"position_delta_{i}"] = np.array([0, 0])
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
        description = build_prompt(relative_path)
        test_list.append((condition_list, description))
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
        res = res.images[0].resize(image.size)
        file_path = os.path.join(
            save_path, f"{file_name}_{'|'.join(condition_type)}_{i}.jpg"
        )
        res.save(file_path)


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
    # else:
    #     dataset = load_dataset(
    #         "webdataset",
    #         data_files={"train": dataset_config["urls"]},
    #         split="train",
    #         cache_dir="cache/t2i2m",
    #         num_proc=32,
    #     )
    #     dataset = ImageMultiConditionDataset(
    #         dataset,
    #         condition_size=dataset_config["condition_size"],
    #         target_size=dataset_config["target_size"],
    #         condition_type=training_config["condition_type"],
    #         drop_text_prob=dataset_config["drop_text_prob"],
    #         drop_image_prob=dataset_config["drop_image_prob"],
    #         position_scale=dataset_config.get("position_scale", 1.0),
    #     )

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
