from pydantic import BaseModel, ConfigDict
from typing import List, Literal, Optional


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_type: str = "BaseDataset"
    train_class_0_paths: List[str]
    train_class_1_paths: List[str]
    val_class_0_paths: List[str]
    val_class_1_paths: List[str]
    img_height: int = 256
    img_width: int = 256


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_type: str = "ConvNeXt"
    num_classes: int = 2
    pretrained_path: Optional[str] = None
    freeze_backbone: bool = False


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    optimizer_type: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)

class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trainer_type: Literal["LightningTrainer"] = "LightningTrainer"
    save_dir: str
    batch_size: int = 8
    epochs: int = 100
    save_every_n_epochs: int = 5
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    num_workers: int = 4
    persistent_workers: bool = True
    log_every_n_steps: int = 10
    check_val_every_n_epoch: int = 1
    accelerator: str = "auto"
    devices: int = 1
    precision:  str = "16-mixed"

class FullConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    seed: int = 42
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    loss: str = "cross_entropy"


