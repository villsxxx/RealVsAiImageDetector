import json
import random
import numpy as np
import torch
from datetime import datetime
import os
import glob
import Datasets as datasets_module
import Models as models_module
import Trainers as trainer_module
from .config import FullConfig

class ConfigParser:
    def __init__(self, config_dict):
        self.config = FullConfig(**config_dict)
        self._set_seed(self.config.seed)
        self._init_save_dirs()

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_save_dirs(self):
        base_save_dir = self.config.trainer.save_dir
        name = self.config.name
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        self.run_dir = os.path.join(base_save_dir, name, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        self.model_dir = os.path.join(self.run_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
            f.write(self.config.model_dump_json(indent=2))

    def _expand_glob(self, patterns):
        files = []
        for pattern in patterns:
            expanded = glob.glob(pattern)
            if not expanded:
                raise ValueError(f"Pattern '{pattern}' did not match any files")
            files.extend(expanded)
        return files

    def init_dataset(self, split="train"):
        ds_cfg = self.config.dataset
        cls_ = getattr(datasets_module, ds_cfg.dataset_type)

        if split == "train":
            raw_class_0_paths = ds_cfg.train_class_0_paths
            raw_class_1_paths = ds_cfg.train_class_1_paths
        else:
            raw_class_0_paths = ds_cfg.val_class_0_paths
            raw_class_1_paths = ds_cfg.val_class_1_paths

        class_0_paths = self._expand_glob(raw_class_0_paths)
        class_1_paths = self._expand_glob(raw_class_1_paths)

        pairs = []
        for p in class_0_paths:
            pairs.append((p, 0))
        for p in class_1_paths:
            pairs.append((p, 1))

        params = ds_cfg.model_dump(
            exclude={'dataset_type',
                     'train_class_0_paths', 'train_class_1_paths',
                     'val_class_0_paths', 'val_class_1_paths'}
        )
        params['pairs'] = pairs
        params['is_train'] = (split == 'train')
        return cls_(**params)

    def init_model(self):
        model_cfg = self.config.model
        cls_ = getattr(models_module, model_cfg.model_type)
        params = model_cfg.model_dump(exclude={'model_type'})
        return cls_(**params)

    def get_optimizer_config(self):
        opt_cfg = self.config.optimizer
        config = opt_cfg.model_dump(exclude={'optimizer_type'})
        return {'type': opt_cfg.optimizer_type, 'args': config}

    def get_loss(self):
        loss_name = self.config.loss
        if loss_name == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif loss_name == "bce":
            return torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

    def init_trainer(self, model, train_dataset, val_dataset, criterion):
        trainer_cfg = self.config.trainer
        trainer_type = trainer_cfg.trainer_type
        cls_ = getattr(trainer_module, trainer_type)

        trainer_params = trainer_cfg.model_dump(exclude={'trainer_type', 'save_dir'})
        trainer_params.update({
            'model': model,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'criterion': criterion,
            'model_dir': self.model_dir,
            'seed': self.config.seed,
            'optimizer_config': self.get_optimizer_config(),
        })
        return cls_(**trainer_params)