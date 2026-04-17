# main.py
"""
Script principal para clasificación binaria de mamografías BI-RADS.
Ejecuta localmente en Linux.
"""

import os
import torch
from pathlib import Path

# Configuración global
CONFIG = {
    # Directorios (ajustados para local)
    "database1_root": str(Path(__file__).parent / "database1"),  # Entrenamiento/validación
    "database2_root": str(Path(__file__).parent / "database2"),  # Prueba
    "checkpoint_dir": str(Path(__file__).parent / "checkpoints"),

    # Hiperparámetros de entrenamiento
    "epochs":          50,
    "batch_size":      32,
    "learning_rate":   1e-4,
    "weight_decay":    1e-4,
    "val_fraction":    0.20,
    "num_workers":     0,  # Cambiado a 0 para evitar problemas en CPU
    "seed":            42,

    # Warm-up
    "warmup_epochs":   3,

    # Early stopping
    "patience":        8,

    # Evaluación
    "threshold":       0.5,
}

def setup_environment():
    """Configura el entorno local"""

    print("\n" + "="*70)
    print("CONFIGURACIÓN DEL ENTORNO LOCAL")
    print("="*70)

    # Crear directorio de checkpoints
    Path(CONFIG["checkpoint_dir"]).mkdir(exist_ok=True, parents=True)
    print(f"\n✅ Directorio de checkpoints: {CONFIG['checkpoint_dir']}")

    # Verificar GPU
    if torch.cuda.is_available():
        print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU no disponible, se usará CPU (será más lento)")

    print("\n" + "="*70)

def main():
    """Función principal que orquesta todo el pipeline"""

    print("\n" + "="*70)
    print("CLASIFICACIÓN BINARIA DE MAMOGRAFÍAS BI-RADS")
    print("="*70)
    print("\nEste script realiza:")
    print("  1. Configuración del entorno local")
    print("  2. Entrenamiento del modelo ResNet-50")
    print("  3. Evaluación en conjunto de prueba")
    print("="*70)

    # Configurar entorno
    setup_environment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # Verificar que los directorios existen
    if not Path(CONFIG["database1_root"]).exists():
        print(f"\n❌ ERROR: No se encuentra el directorio {CONFIG['database1_root']}")
        print("Asegúrate de que las carpetas database1/ y database2/ estén en el mismo directorio que este script.")
        return None, None

    # Importar módulos
    from data_loader import get_train_val_loaders
    from train import run_training
    from evaluate import run_evaluation

    # Data
    print("\n📂 Cargando datos...")
    train_loader, val_loader, class_weights = get_train_val_loaders(
        database1_root=CONFIG["database1_root"],
        val_fraction=CONFIG["val_fraction"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        seed=CONFIG["seed"],
    )

    # Entrenamiento
    model, checkpoint_path = run_training(CONFIG, train_loader, val_loader, class_weights, device)

    # Evaluación
    if model is not None:
        results = run_evaluation(CONFIG, checkpoint_path)

    print("\n" + "="*70)
    print("🏁 PROCESO COMPLETADO")
    print("="*70)

if __name__ == "__main__":
    main()