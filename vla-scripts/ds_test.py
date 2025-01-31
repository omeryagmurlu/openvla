"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy
import numpy as np

from prismatic.vla.datasets.datasets import RLDSDataset

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class VLA:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2", default_unnorm=None) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.default_unnorm = default_unnorm

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)
        
        # trigger compile if there is one

        prompt = get_openvla_prompt("test inst", self.openvla_path)
        inputs = self.processor(prompt, Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key=self.default_unnorm, do_sample=False)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        # Parse payload components
        image, instruction = payload["image"], payload["instruction"]
        unnorm_key = payload.get("unnorm_key", self.default_unnorm)

        # Run VLA Inference
        prompt = get_openvla_prompt(instruction, self.openvla_path)
        inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

        return action

@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = "openvla/openvla-7b"               # HF Hub Path (or path to local run directory)
    # openvla_path: Union[str, Path] = "/home/reuss/code/openvla/runs/openvla-7b+kit_irl_real_kitchen_lang+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug"               # HF Hub Path (or path to local run directory)
    default_unnorm = "bridge_orig"
    # default_unnorm = "kit_irl_real_kitchen_lang"
    # default_unnorm = None

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    inst = VLA(cfg.openvla_path, default_unnorm=cfg.default_unnorm)

    vla_dataset = RLDSDataset(
        "/home/reuss/tensorflow_datasets",
        "bridge_orig",
        None,
        resize_resolution=(224,224),
        shuffle_buffer_size=100,
        image_aug=None,
    )


    it = iter(vla_dataset.dataset)
    sample = next(it)

    payload = {}
    payload['image'] = sample['observation']['image_primary'][0].numpy()
    payload['instruction'] = sample['task']['language_instruction'].numpy().decode()

    action = inst.predict_action(payload)

    gt_action = sample['action'].numpy()

    print(torch.nn.functional.mse_loss(action, gt_action))

    0

if __name__ == "__main__":
    deploy()
