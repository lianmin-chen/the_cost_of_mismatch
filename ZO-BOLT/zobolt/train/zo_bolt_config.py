
import warnings
from dataclasses import dataclass, field

from transformers import TrainingArguments
from trl import GRPOConfig, SFTConfig
@dataclass
class ZOBOLTConfig(GRPOConfig):
    # Training
    trainer: str = "zo"
    ## options
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo: zeroth-order training
    
    # Gaussian finite-difference radius for ZoAR.
    zo_eps: float = 1e-3

    # R-AdaZO moments and ZoAR history.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    num_histories: int = 4 # zoar history
    loss_imp: str = "liger" # original, flce, cce, liger
