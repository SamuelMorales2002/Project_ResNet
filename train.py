# train.py
"""
Módulo para entrenamiento del modelo.
"""

import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from model import build_model

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.float32).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.float32).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total

def run_training(config, train_loader, val_loader, class_weights, device):
    """Ejecuta el pipeline completo de entrenamiento"""

    print("\n" + "="*70)
    print("INICIANDO ENTRENAMIENTO")
    print("="*70)

    # Fijar semillas para reproducibilidad
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    # ------------------------------------------------------------------
    # Modelo - Fase 1: backbone congelado
    # ------------------------------------------------------------------
    print("\n🏗️  Construyendo modelo (fase warm-up: backbone congelado)...")
    model = build_model(pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # Pérdida ponderada
    pos_weight = (class_weights[1] / class_weights[0]).unsqueeze(0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    best_ckpt = ckpt_dir / "best_model.pt"

    # ------------------------------------------------------------------
    # Bucle de entrenamiento
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n🚀 Comenzando entrenamiento...\n")
    print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Time")
    print("-" * 65)

    for epoch in range(1, config["epochs"] + 1):

        # Después del warm-up, descongelar backbone
        if epoch == config["warmup_epochs"] + 1:
            print("\n" + "🔓" * 20)
            print("[INFO] Warm-up completado — descongelando backbone")
            print("🔓" * 20 + "\n")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                          patience=3)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)
        elapsed = time.time() - t0

        scheduler.step(val_loss)

        print(f"{epoch:04d}  | {train_loss:.4f}    | {train_acc:.3f}     | {val_loss:.4f}   | {val_acc:.3f}   | {elapsed:.1f}s")

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": config,
                },
                best_ckpt,
            )
            print(f"  ✨ -> Mejor modelo guardado (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n⏹️  Early stopping en epoch {epoch} "
                      f"(sin mejora por {config['patience']} epochs)")
                break

    print(f"\n✅ Entrenamiento completado. Mejor val_loss: {best_val_loss:.4f}")
    print(f"📁 Checkpoint guardado en: {best_ckpt}")

    return model, best_ckpt