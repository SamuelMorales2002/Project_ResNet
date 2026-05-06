# evaluate.py
"""
Módulo para evaluación del modelo.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

@torch.no_grad()
def predict(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_probs  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= threshold).astype(int)
    return all_labels, all_preds, all_probs


def find_best_threshold(model, val_loader, device):
    """Encuentra el mejor threshold para maximizar F1 en la clase riesgo usando el conjunto de validación"""
    print("\n🔍 Buscando el mejor threshold en validación...")
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"✅ Mejor threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benigno", "Riesgo"],
        yticklabels=["Benigno", "Riesgo"],
        ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión — Conjunto de Prueba")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Matriz de confusión guardada en: {save_path}")


def run_evaluation(config, checkpoint_path=None, val_loader=None):
    """Evalúa el modelo guardado en el conjunto de prueba"""

    print("\n" + "="*70)
    print("EVALUACIÓN EN CONJUNTO DE PRUEBA")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    if checkpoint_path is None:
        checkpoint_path = Path(config["checkpoint_dir"]) / "best_model.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"❌ No se encuentra el checkpoint en {checkpoint_path}")
        print("Primero ejecuta el entrenamiento.")
        return None

    # Verificar datos de prueba
    if not Path(config["database2_root"]).exists():
        print(f"\n❌ ERROR: No se encuentra el directorio {config['database2_root']}")
        print("Por favor, verifica la ruta de tus datos de prueba.")
        return None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    from data_loader import get_test_loader
    print("\n📂 Cargando datos de prueba...")
    test_loader = get_test_loader(
        database2_root=config["database2_root"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # ------------------------------------------------------------------
    # Modelo
    # ------------------------------------------------------------------
    from model import build_model, load_checkpoint
    print("\n🏗️  Cargando modelo...")
    model = build_model(pretrained=False)
    model = model.to(device)
    ckpt = load_checkpoint(model, str(checkpoint_path), device)
    print(f"✅ Checkpoint cargado - Época: {ckpt.get('epoch', '?')}, "
          f"Val Loss: {ckpt.get('val_loss', '?'):.4f}")

    # ------------------------------------------------------------------
    # Ajuste de threshold
    # ------------------------------------------------------------------
    if val_loader is not None:
        threshold = find_best_threshold(model, val_loader, device)
    else:
        threshold = config["threshold"]

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------
    print("\n🔮 Realizando predicciones...")
    labels, preds, probs = predict(
        model, test_loader, device, threshold)

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------
    acc       = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)
    auc       = roc_auc_score(labels, probs)
    cm        = confusion_matrix(labels, preds)

    print("\n" + "="*50)
    print("📈 RESULTADOS EN CONJUNTO DE PRUEBA")
    print("="*50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}  (clase riesgo)")
    print(f"  Recall    : {recall:.4f}  (clase riesgo) ← prioritaria")
    print(f"  F1-score  : {f1:.4f}  (clase riesgo)")
    print(f"  AUC-ROC   : {auc:.4f}")
    print("="*50)
    print("\n📋 Reporte de clasificación:")
    print(classification_report(labels, preds,
                                target_names=["Benigno", "Riesgo"],
                                zero_division=0))

    # Guardar matriz de confusión
    plot_confusion_matrix(cm, str(Path(config["checkpoint_dir"]) / "confusion_matrix.png"))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm
    }