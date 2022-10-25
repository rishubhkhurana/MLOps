import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import sys
sys.path.append(root)

# importing libraries for the machine learning 
from typing import List, Optional, Tuple, Dict
import hydra
import torch
from PIL import Image
import pytorch_lightning as pl
from omegaconf import DictConfig
import torchvision.transforms as T
from src import utils
import gradio as gr
import numpy as np

log = utils.get_pylogger(__name__)

CIFAR10_CLASSES = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="demo.yaml")
def main(cfg: DictConfig) -> Tuple[dict, dict]:

    log.info(f"Instantiating model <{cfg.model._target_}>")
    # model: LightningModule = hydra.utils.instantiate(cfg.model)
    # model_state_dict = torch.load(cfg.best_model_path)['state_dict']
    # model.load_state_dict(model_state_dict)
    # scripted = torch.jit.script(model)
    model = torch.jit.load(cfg.best_model_path)
    
    def predict(inp_img: Image) -> Dict[str, float]:
        image = T.ToTensor()(inp_img).unsqueeze(0)
        probs = model.forward_jit(image)[0]
        result = {CIFAR10_CLASSES[i]: prb.item() for i, prb in enumerate(probs)}
        return result

    gr.Interface(fn = predict,  inputs = gr.Image(type='pil'), outputs = gr.Label(num_top_classes=5)).launch(share=True)


if __name__ == "__main__":
    main()