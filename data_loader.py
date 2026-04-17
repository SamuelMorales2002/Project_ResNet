# data_loader.py
"""
Módulo para carga y preparación de datos de mamografías BI-RADS.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from PIL import Image

# Mapeo de carpetas a etiquetas binarias
BENIGN_FOLDERS = {"birads1"}
RISK_FOLDERS   = {"birads3", "birads4", "birads5"}

# Transformaciones
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class MammogramDataset(Dataset):
    """
    Carga imágenes de carpetas birads1/, birads3/, birads4/, birads5/
    Etiquetas binarias: 0 -> benigno (birads1), 1 -> riesgo (birads3/4/5)
    """

    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            name = folder.name.lower()
            if name in BENIGN_FOLDERS:
                label = 0
            elif name in RISK_FOLDERS:
                label = 1
            else:
                continue

            for img_file in sorted(folder.iterdir()):
                if img_file.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((img_file, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No se encontraron imágenes en {root}. "
                             f"Se esperan subcarpetas: birads1, birads3, birads4, birads5.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error cargando imagen {img_path}: {e}")
            raise RuntimeError(f"Error cargando imagen {img_path}: {e}")
        return image, label

    def class_counts(self) -> dict:
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        counts = self.class_counts()
        total = sum(counts.values())
        weights = torch.tensor(
            [total / (2 * counts[c]) for c in [0, 1]], dtype=torch.float32
        )
        return weights

    def sample_weights(self) -> List[float]:
        cw = self.class_weights().tolist()
        return [cw[label] for _, label in self.samples]

def get_train_val_loaders(
    database1_root: str,
    val_fraction: float = 0.20,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Divide database1 en entrenamiento y validación (estratificado por clase)
    """
    rng = np.random.default_rng(seed)

    # Cargar dataset completo para obtener etiquetas
    full_dataset = MammogramDataset(database1_root, transform=None)
    labels = [lbl for _, lbl in full_dataset.samples]

    indices_0 = np.where(np.array(labels) == 0)[0]
    indices_1 = np.where(np.array(labels) == 1)[0]

    def split_indices(idx_array):
        rng.shuffle(idx_array)
        val_n = max(1, int(len(idx_array) * val_fraction))
        return idx_array[val_n:].tolist(), idx_array[:val_n].tolist()

    train_idx_0, val_idx_0 = split_indices(indices_0)
    train_idx_1, val_idx_1 = split_indices(indices_1)

    train_indices = train_idx_0 + train_idx_1
    val_indices   = val_idx_0 + val_idx_1

    # Datasets con transforms apropiados
    train_dataset = MammogramDataset(database1_root, transform=TRAIN_TRANSFORMS)
    val_dataset   = MammogramDataset(database1_root, transform=EVAL_TRANSFORMS)

    train_subset = Subset(train_dataset, train_indices)
    val_subset   = Subset(val_dataset, val_indices)

    # WeightedRandomSampler para sobremuestrear clase minoritaria
    all_sample_weights = train_dataset.sample_weights()
    train_weights = [all_sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_indices),
        replacement=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_weights = train_dataset.class_weights()

    # Información del split
    n_train_benign = sum(labels[i] == 0 for i in train_indices)
    n_train_risk   = sum(labels[i] == 1 for i in train_indices)
    n_val_benign   = sum(labels[i] == 0 for i in val_indices)
    n_val_risk     = sum(labels[i] == 1 for i in val_indices)

    print(f"\n📊 División de datos:")
    print(f"   Train -> benigno: {n_train_benign}, riesgo: {n_train_risk}")
    print(f"   Val   -> benigno: {n_val_benign},   riesgo: {n_val_risk}")
    print(f"   Pesos de clase -> benigno: {class_weights[0]:.3f}, riesgo: {class_weights[1]:.3f}")

    return train_loader, val_loader, class_weights


def get_test_loader(
    database2_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
) -> DataLoader:
    """Devuelve un DataLoader para el conjunto de prueba"""
    test_dataset = MammogramDataset(database2_root, transform=EVAL_TRANSFORMS)
    counts = test_dataset.class_counts()
    print(f"\n📊 Conjunto de prueba:")
    print(f"   benigno: {counts[0]}, riesgo: {counts[1]}")
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )