# Clasificación Binaria de Mamografías BI-RADS

Este proyecto implementa un clasificador binario para mamografías usando ResNet-50, diferenciando entre casos benignos (BI-RADS 1) y de riesgo (BI-RADS 3, 4, 5).

## Estructura del Proyecto

- `main.py`: Script principal para ejecutar el pipeline completo.
- `data_loader.py`: Módulo para carga y preprocesamiento de datos.
- `model.py`: Definición del modelo ResNet-50.
- `train.py`: Funciones de entrenamiento.
- `evaluate.py`: Funciones de evaluación y métricas.
- `requirements.txt`: Dependencias de Python.
- `database1/`: Datos de entrenamiento y validación.
- `database2/`: Datos de prueba.
- `checkpoints/`: Modelos guardados (se crea automáticamente).

## Instalación

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Asegúrate de que las carpetas `database1/` y `database2/` contengan las subcarpetas `birads1/`, `birads3/`, `birads4/`, `birads5/` con las imágenes.

## Uso

Ejecuta el script principal:
```bash
python main.py
```

Esto realizará:
- Carga y división de datos.
- Entrenamiento del modelo con early stopping.
- Evaluación en el conjunto de prueba.
- Guardado de métricas y matriz de confusión.

## Configuración

Edita `CONFIG` en `main.py` para ajustar hiperparámetros, rutas, etc.

## Requisitos

- Python 3.8+
- PyTorch con CUDA (opcional, pero recomendado para GPU)