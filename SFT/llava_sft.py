#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SFT adapter for llava-format datasets with optional custom validation dataset."""

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import toml
import torch.utils.data
from cosmos_reason2_utils.text import create_conversation
from cosmos_reason2_utils.vision import PIXELS_PER_TOKEN, VisionConfig
from cosmos_rl.utils.logging import logger


class CustomDatasetConfig(pydantic.BaseModel):
    annotation_path: str = pydantic.Field()
    """Training annotation path."""

    media_path: str = pydantic.Field(default="")
    """Training media root path."""

    # Optional custom validation dataset paths.
    val_annotation_path: str = pydantic.Field(default="")
    """Validation annotation path."""

    val_media_path: str = pydantic.Field(default="")
    """Validation media root path. If empty, media_path is used."""

    system_prompt: str = pydantic.Field(default="")
    """System prompt for post-training."""

    val_system_prompt: str = pydantic.Field(default="")
    """Validation system prompt. If empty, system_prompt is used."""


class CustomConfig(pydantic.BaseModel):
    dataset: CustomDatasetConfig = pydantic.Field()
    """Dataset config."""

    vision: VisionConfig = pydantic.Field(
        default=VisionConfig(
            fps=1,
        )
    )
    """Vision processor config."""


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotation_path: str,
        media_path: str,
        system_prompt: str,
        vision_kwargs: dict[str, Any],
    ):
        with open(annotation_path, encoding="utf-8") as f:
            self.annotation = json.load(f)
        self.annotation_path = annotation_path
        self.media_path = media_path
        self.system_prompt = system_prompt
        self.vision_kwargs = vision_kwargs

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx: int) -> list[dict]:
        try:
            sample = self.annotation[idx]
            user_prompt = sample["conversations"][0]["value"]
            response = sample["conversations"][1]["value"]
        except (KeyError, IndexError) as e:
            logger.error(
                f"Missing required field in sample at index {idx} "
                f"(annotation={self.annotation_path}): {e}"
            )
            logger.error(f"Sample keys: {list(sample.keys())}")
            raise

        images = sample.get("image", None) or sample.get("images", None)
        if images and isinstance(images, str):
            images = [images]
        videos = sample.get("video", None)
        if videos and isinstance(videos, str):
            videos = [videos]

        images = images or []
        videos = videos or []

        if self.media_path != "":
            if images:
                images = [os.path.join(self.media_path, img) for img in images]
            if videos:
                videos = [os.path.join(self.media_path, vid) for vid in videos]

        for i, image in enumerate(images):
            try:
                with open(image, "rb") as f:
                    images[i] = base64.b64encode(f.read())
            except (OSError, FileNotFoundError) as e:
                logger.error(
                    f"Failed to read image file at sample index {idx}, "
                    f"image index {i}: {e}"
                )
                raise

        # Remove image/video tags from prompt; media is passed separately.
        user_prompt = re.sub(r"(\n)?</?(image|video)>(\n)?", "", user_prompt)

        conversations = create_conversation(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response=response,
            images=images,
            videos=videos,
            vision_kwargs=self.vision_kwargs,
        )
        return conversations


def build_dataset(
    *,
    annotation_path: str,
    media_path: str,
    system_prompt: str,
    vision_kwargs: dict[str, Any],
) -> CustomDataset:
    dataset = CustomDataset(
        annotation_path=annotation_path,
        media_path=media_path,
        system_prompt=system_prompt,
        vision_kwargs=vision_kwargs,
    )
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty: {annotation_path}")
    # Early fail-fast validation of schema/path issues.
    dataset[0]
    return dataset


def wandb_validation_alias_logger(report_data: dict[str, Any], step: int) -> None:
    """Alias cosmos-rl validation metric to validation_loss in wandb."""
    if "val/avg_loss" not in report_data:
        return
    try:
        import wandb  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.log({"validation_loss": report_data["val/avg_loss"]}, step=step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    args = parser.parse_known_args()[0]

    with open(args.config, encoding="utf-8") as f:
        config_kwargs = toml.load(f)
    config = cosmos_rl.policy.config.Config.from_dict(config_kwargs)
    custom_config = CustomConfig.model_validate(config_kwargs["custom"])

    custom_config.vision.total_pixels = int(
        config.policy.model_max_length * PIXELS_PER_TOKEN * 0.9
    )
    vision_kwargs = custom_config.vision.model_dump(exclude_none=True)

    role = os.environ.get("COSMOS_ROLE")
    is_controller = role == "Controller"
    if is_controller:
        output_dir = Path(config.train.output_dir).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save resolved config.
        config_dump = config.model_dump()
        config_dump["custom"] = custom_config.model_dump()
        config_path = output_dir / "config.toml"
        config_path.write_text(toml.dumps(config_dump), encoding="utf-8")
        logger.info(f"Saved config to {config_path}")

    train_dataset = build_dataset(
        annotation_path=custom_config.dataset.annotation_path,
        media_path=custom_config.dataset.media_path,
        system_prompt=custom_config.dataset.system_prompt,
        vision_kwargs=vision_kwargs,
    )
    logger.info(f"Loaded training dataset: size={len(train_dataset)}")

    val_dataset: Optional[CustomDataset] = None
    if custom_config.dataset.val_annotation_path:
        val_dataset = build_dataset(
            annotation_path=custom_config.dataset.val_annotation_path,
            media_path=custom_config.dataset.val_media_path
            or custom_config.dataset.media_path,
            system_prompt=custom_config.dataset.val_system_prompt
            or custom_config.dataset.system_prompt,
            vision_kwargs=vision_kwargs,
        )
        logger.info(f"Loaded validation dataset: size={len(val_dataset)}")
    elif config.validation.enable:
        logger.warning(
            "validation.enable=true but custom.dataset.val_annotation_path is empty. "
            "Training will run without your custom validation dataset."
        )

    if val_dataset is not None and not config.validation.enable:
        logger.warning(
            "Validation dataset is provided but validation.enable=false. "
            "Set validation.enable=true to compute/log validation metrics."
        )

    cosmos_rl.launcher.worker_entry.main(
        dataset=train_dataset,
        val_dataset=val_dataset,
        custom_logger_fns=[wandb_validation_alias_logger],
    )
