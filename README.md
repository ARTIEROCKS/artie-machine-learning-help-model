# HelpModel: LSTM + Atención explicable con DVC y TensorBoard

## 1. Descripción
Proyecto para entrenar un modelo LSTM (opcionalmente bidireccional y con mecanismo de atención explicable) que predice `request_help` en secuencias temporales educativas. Orquestación con **DVC**, modelado con **TensorFlow/Keras**, registro de métricas y evolución con **TensorBoard** y exportación de un submodelo de atención para análisis explicativo.

## 2. Flujo de procesamiento
1. Descarga / generación de datos (`download` – opcional según tu pipeline).
2. Transformación a CSV (`transformation`).
3. Análisis exploratorio y SHAP (`dataanalysis`).
4. Selección de características (`featureselection`).
5. Entrenamiento del modelo LSTM con o sin atención (`train`).
6. Evaluación y extracción de predicciones + pesos de atención (`test_train`).

## 3. Estructura principal
```
├── dvc.yaml                # Definición de stages DVC
├── params.yaml             # Hiperparámetros y configuración
├── src/
│   ├── train.py            # Entrenamiento principal
│   ├── test_train.py       # Inferencia + extracción atención
│   ├── keras_custom_layers.py # Funciones/capas personalizadas serializables
│   ├── data_analysis.py, featureselection.py, formatcsv*.py, download.py
├── model/                  # Modelos guardados (.keras)
├── metrics/                # Métricas, predicciones, atención, plots DVC
├── images/                 # Figuras (arquitectura, predicciones, etc.)
├── logs/                   # Eventos TensorBoard (train, validation, learning_rate)
└── requirements.txt
```

## 4. Requisitos
- Python 3.11+
- macOS ARM (ejemplo) con soporte Metal para acelerar TensorFlow.

Instalación base:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Para GPU (macOS Metal) si faltase:
```bash
pip install tensorflow-macos tensorflow-metal
```

## 5. Parámetros clave (`params.yaml` – sección `model`)
- `mask_value`: valor de padding/masking.
- `percentage_train_size`: % de secuencias para entrenamiento.
- `lstm_units`: tamaño de unidades LSTM.
- `return_sequences`: si la capa LSTM final devuelve secuencias completas.
- `second_lstm_layer`: añade una LSTM previa adicional.
- `use_dropouts` + `dropout_value`: regularización.
- `use_bidirectional`: activa Bidirectional LSTM.
- `use_attention`: activa mecanismo de atención explicable.
- `initial_learning_rate`: LR inicial Adam.
- `training_epochs`, `training_batch_size`.
- `training_class_weights`: activar cálculo de class weights.
- Callbacks: `training_early_stopping_patience`, `training_reduce_lr_patience`, `training_reduce_lr_factor`.
- `show_summary`: imprime summary del modelo.

Cambia valores y luego ejecuta:
```bash
dvc repro train
```

## 6. Ejecución de la pipeline con DVC
Mostrar grafo:
```bash
dvc dag
```
Reproducir todo:
```bash
dvc repro
```
Solo entrenamiento:
```bash
dvc repro train
```
Ver métricas:
```bash
dvc metrics show
```
Ver plots:
```bash
dvc plots show
```

## 7. Ejecución manual del entrenamiento (fuera de DVC)
```bash
python src/train.py \
  --params-file params.yaml \
  --input-csv-file data/featureselection.csv \
  --output-model-file model/help_model.keras \
  --plots-file-name metrics/plots.csv \
  --metrics-file-name metrics/scores.json \
  --use-gpu \
  --output-dir images
```
Genera también (si `use_attention: true`):
- `model/help_model_attention.keras` (submodelo de atención)
- `images/model.png`
- `metrics/scores.json`, `metrics/plots.csv`

## 8. Test / inferencia y extracción de atención
```bash
python src/test_train.py params.yaml model/help_model.keras data/featureselection.csv \
  --output_dir images --use_gpu --test-sequences 15
```
Produce:
- `metrics/test_predictions.csv` (predicciones por paso)
- `metrics/test_attention.csv` (pesos de atención alineados)
- Figuras opcionales en `images/` (desactivar con `--no-viz`).

## 9. Mecanismo de atención
Cuando `use_attention` está activo:
1. Dense(1, tanh) produce `attention_score` por paso.
2. Se comprime eje final (squeeze) -> vector (batch, time_steps).
3. Se genera máscara (1 válido / 0 padding) usando `mask_value`.
4. Se aplican -∞ a pasos enmascarados para que Softmax los anule.
5. Softmax normaliza en ejes temporales válidos → `attention_weights`.
6. Se aplica multiplicación elemento a elemento (sin reducir) sobre la secuencia.
7. Capa final Dense(1, sigmoid) genera probabilidades por paso.

Submodelo de atención (`help_model_attention.keras`) devuelve solo `attention_weights` para el mismo input, permitiendo auditoría explicable.

## 10. Extracción programática de pesos de atención
```python
import tensorflow as tf
from keras_custom_layers import (compute_mask_layer, squeeze_last_axis_func,
    mask_attention_scores_func, apply_attention_func, AttentionLayer, MaskedRepeatVector)

model = tf.keras.models.load_model(
    "model/help_model.keras", compile=False,
    custom_objects={
        "compute_mask_layer": compute_mask_layer,
        "squeeze_last_axis_func": squeeze_last_axis_func,
        "mask_attention_scores_func": mask_attention_scores_func,
        "apply_attention_func": apply_attention_func,
        "AttentionLayer": AttentionLayer,
        "MaskedRepeatVector": MaskedRepeatVector
    }
)
att_model = tf.keras.models.load_model(
    "model/help_model_attention.keras", compile=False,
    custom_objects={
        "compute_mask_layer": compute_mask_layer,
        "squeeze_last_axis_func": squeeze_last_axis_func,
        "mask_attention_scores_func": mask_attention_scores_func,
        "apply_attention_func": apply_attention_func
    }
)
# batch_x shape: (batch, time_steps, features)
attention_weights = att_model.predict(batch_x)
```

## 11. TensorBoard
Logs generados automáticamente en `logs/` (subdirectorios train, validation, learning_rate). Ejecutar:
```bash
tensorboard --logdir logs --port 6006
```
Abre `http://localhost:6006` para visualizar curvas de loss, métricas y LR.

## 12. Métricas y artefactos
- `metrics/scores.json`: métricas finales (loss, binary_accuracy, precision, recall, auc, etc.).
- `metrics/plots.csv`: histórico por epoch (para DVC plots).
- `metrics/test_predictions.csv`: inferencia paso a paso (etiqueta real + predicción).
- `metrics/test_attention.csv`: matriz de pesos de atención.
- `model/help_model.keras`: modelo completo.
- `model/help_model_attention.keras`: submodelo de atención.

## 13. Uso de GPU
Activar con `--use-gpu` (train y test). Se listan dispositivos y se fuerza el uso de GPU si está disponible (Metal en macOS). Si ves `memory 0 MB` en Metal es normal (gestión diferida). Para ver que realmente ejecuta en GPU puedes añadir:
```python
print(tf.config.list_logical_devices('GPU'))
```

## 14. Serialización segura (evitando lambdas inline)
Las funciones usadas en capas Lambda están definidas y registradas en `keras_custom_layers.py` con `@tf.keras.utils.register_keras_serializable` para permitir carga sin `safe_mode=False`. Evita errores como:
`Could not locate function 'func'`.

## 15. Problemas comunes y soluciones
| Problema | Causa | Solución |
|----------|-------|----------|
| `IndexError: list index out of range` al importar `train.py` | Ejecución accidental al importar | Asegurar bloque `if __name__ == "__main__":` (implementado). |
| Error deserializando Lambda | Función no registrada | Usar/añadir decorador `@register_keras_serializable`. |
| Pesos no cargan (0 variables) | Falta `custom_objects` o mismatch | Pasar diccionario con funciones registradas. |
| GPU ignorada | Falta flag o instalación | Añadir `--use-gpu` e instalar `tensorflow-metal`. |

## 16. Añadir un nuevo hiperparámetro rastreable
1. Añadir en `params.yaml`.
2. Consumirlo en `train.py` / `test_train.py` (argparse o lectura YAML).
3. Añadirlo en `dvc.yaml` bajo `stages.train.params`.
4. Ejecutar `dvc repro`.

## 17. Limpieza y regeneración
```bash
rm -f model/help_model*.keras
rm -f metrics/*.json metrics/*.csv
dvc repro train
```

## 18. Visualización de secuencias
`test_train.py` genera figuras de predicción y (opcionalmente) importancia/atención por secuencia. Desactiva con `--no-viz` para ejecuciones batch.

## 19. Próximas mejoras sugeridas
- Exportación ONNX.
- SHAP sobre embeddings intermedios + atención.
- Validación temporal (walk-forward).
- Paquete Docker reproducible.

## 20. Comandos rápidos
```bash
# Entrenar (pipeline)
dvc repro train

# Entrenar manual
python src/train.py --params-file params.yaml --input-csv-file data/featureselection.csv \
  --output-model-file model/help_model.keras --plots-file-name metrics/plots.csv \
  --metrics-file-name metrics/scores.json --use-gpu --output-dir images

# Test / inferencia
python src/test_train.py params.yaml model/help_model.keras data/featureselection.csv \
  --output_dir images --use_gpu --test-sequences 10 --no-viz

# TensorBoard
tensorboard --logdir logs

# Métricas DVC
dvc metrics show
# Plots DVC
dvc plots show
```

## 21. Licencia
(Completar según corresponda.)

---
Si necesitas ampliar este README (por ejemplo, sección de troubleshooting más profunda o guía de despliegue), indícalo y lo ajustamos.

