# HelpModel: Explainable LSTM + Attention with DVC and TensorBoard

## 1. Overview
HelpModel trains an LSTM (optionally bidirectional and with an intrinsically explainable attention mechanism) to predict `request_help` events in educational temporal interaction sequences. The project integrates:
- TensorFlow/Keras for modeling
- A custom attention mechanism (serializable, mask-aware)
- DVC for reproducible data & experiment pipeline
- TensorBoard for metric and training dynamics visualization
- Export of a standalone attention submodel for post‑hoc explainability

## 2. End-to-End Pipeline (DVC Stages)
1. (Optional) `download`: Download or generate raw source data.
2. `transformation`: Convert raw sources into unified CSV.
3. `dataanalysis`: Exploratory analysis + SHAP computations.
4. `featureselection`: Produce filtered feature set.
5. `train`: Train main LSTM model (with/without attention).
6. `test_train`: Inference, attention extraction, prediction visualization.
7. (Optional added) `attentionheatmap`: Generate a focused heatmap + architecture diagram from a chosen test sequence.

Graph:
```bash
dvc dag
```
Reproduce full pipeline:
```bash
dvc repro
```
Run only training:
```bash
dvc repro train
```

## 3. Repository Structure
```
├── dvc.yaml                 # DVC stages definition
├── dvc.lock                 # Locked versions of stages after repro
├── params.yaml              # Hyperparameters & configuration
├── src/
│   ├── train.py             # Training entrypoint
│   ├── test_train.py        # Inference + attention extraction
│   ├── generate_attention_heatmap.py  # Focused attention visualization (single sequence)
│   ├── keras_custom_layers.py         # Custom serializable attention-related functions
│   ├── data_analysis.py, featureselection.py, download.py, formatcsv*.py
├── model/                   # Saved models (.keras)
├── metrics/                 # Metrics, attention weights, predictions, DVC plots data
├── images/                  # Figures (architecture, heatmaps, sequence prediction plots)
├── logs/                    # TensorBoard event logs
├── data/                    # Processed input data (after feature selection, etc.)
├── requirements.txt
├── README.md
└── LICENSE
```

## 4. Installation
Requirements:
- Python 3.11+
- (macOS example) Optional Metal acceleration for TensorFlow.

Setup:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
GPU (macOS Metal) if missing:
```bash
pip install tensorflow-macos tensorflow-metal
```

## 5. Key Parameters (`params.yaml` > `model`)
| Param | Purpose |
|-------|---------|
| `mask_value` | Padding value used for sequence masking. |
| `percentage_train_size` | % of sequences used for training split. |
| `lstm_units` | Units in principal LSTM layer. |
| `return_sequences` | Whether final LSTM layer returns full sequence. Must be true if attention is enabled. |
| `second_lstm_layer` | Adds an extra LSTM before the final one. |
| `use_dropouts`, `dropout_value` | Regularization settings. |
| `use_bidirectional` | Wrap final LSTM in Bidirectional. |
| `use_attention` | Enables intrinsic attention mechanism. |
| `initial_learning_rate` | Adam initial LR. |
| `training_epochs`, `training_batch_size` | Core training loop config. |
| `training_class_weights` | If true, compute class weights for imbalance. |
| `training_early_stopping_patience` | Early stopping patience epochs. |
| `training_reduce_lr_patience`, `training_reduce_lr_factor` | LR scheduler. |
| `show_summary` | Print model summary. |

Adjust values then:
```bash
dvc repro train
```

## 6. Manual Training Execution
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
Generates (when `use_attention: true`):
- `model/help_model.keras` (full model)
- `model/help_model_attention.keras` (attention submodel returning weights)
- `images/model.png` (general architecture)
- Metrics & per-epoch plots data.

## 7. Inference + Attention Extraction
```bash
python src/test_train.py params.yaml model/help_model.keras data/featureselection.csv \
  --output_dir images --use_gpu --test-sequences 15
```
Outputs:
- `metrics/test_predictions.csv` (per-step real vs predicted probability)
- `metrics/test_attention.csv` (aligned attention weights)
- Sequence prediction & feature importance figures (unless `--no-viz` specified)

## 8. Attention Mechanism (Intrinsic Explainability)
If `use_attention` is enabled:
1. Dense(1, tanh) produces raw attention score per timestep.
2. Squeeze removes singleton dimension → shape (batch, time_steps).
3. A mask is computed from padded steps (any timestep with all features == `mask_value`).
4. Padded positions receive large negative values (effectively -inf) before softmax.
5. Softmax normalizes valid positions → `attention_weights` (probability distribution over real steps).
6. Element-wise weighting of sequence features is produced (preserving time dimension).
7. Final Dense(1, sigmoid) yields per-step probabilities of `request_help`.

Two complementary artifacts:
- Full model: for actual predictions.
- Attention submodel: returns only `attention_weights` for the same inputs, enabling auditing and temporal interpretability.

## 9. Programmatic Attention Retrieval
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

## 10. Focused Attention Heatmap Stage
`generate_attention_heatmap.py` creates:
- A single-sequence attention heatmap (`images/attention_heatmap.png`).
- A model architecture diagram (`images/model_lstm_attention.png`).
If Graphviz/pydot are missing, a placeholder diagram is created (ensuring DVC output consistency). Install real dependencies for a richer diagram:
```bash
pip install pydot graphviz
# macOS system graphviz
brew install graphviz
```

## 11. TensorBoard Integration
Event logs in `logs/` (subdirs: `train`, `validation`, `learning_rate`). Launch:
```bash
tensorboard --logdir logs --port 6006
```
Navigate to http://localhost:6006.

## 12. Metrics & Artifacts Summary
| File | Description |
|------|-------------|
| `metrics/scores.json` | Final aggregate metrics (loss, accuracy, precision, recall, AUC…). |
| `metrics/plots.csv` | Epoch-by-epoch logged metrics for DVC plots. |
| `metrics/test_predictions.csv` | Per-step ground truth vs probability. |
| `metrics/test_attention.csv` | Attention weights aligned to padded sequence length. |
| `images/model.png` | Trained model architecture snapshot. |
| `images/model_lstm_attention.png` | Attention architecture diagram (or placeholder). |
| `images/attention_heatmap.png` | Heatmap for selected sequence. |
| `model/help_model.keras` | Full predictive model. |
| `model/help_model_attention.keras` | Isolated attention submodel. |

## 13. GPU Usage (macOS Metal)
Enable with `--use-gpu`. If GPU devices not listed, ensure `tensorflow-macos` + `tensorflow-metal` are installed. Quick check:
```python
import tensorflow as tf
print(tf.config.list_logical_devices('GPU'))
```

## 14. Safe Serialization (No Inline Lambdas)
All functions used by Lambda layers are defined & registered in `keras_custom_layers.py` using `@tf.keras.utils.register_keras_serializable`. This prevents deserialization errors such as:
```
Could not locate function 'func'
```
By avoiding anonymous inline lambdas, we guarantee portability and stable loading with `compile=False`.

## 15. Common Issues & Resolutions
| Symptom | Cause | Fix |
|---------|-------|-----|
| Deserialization error for Lambda | Missing `@register_keras_serializable` | Ensure custom functions registered (already done). |
| Extra feature columns mismatch | Data pipeline changed post-training | Re-align preprocessing or truncate extra columns (script now truncates when safe). |
| Output file missing in DVC stage | Diagram failed (Graphviz absent) | Fallback placeholder now guarantees file creation. |
| Attention shape mismatch | Sequence dims differ from training | Validate & adjust feature dimension before prediction. |

## 16. Adding a New Hyperparameter
1. Add under `params.yaml`.
2. Consume in `train.py` / `test_train.py`.
3. Reference in `dvc.yaml` stage params.
4. Run `dvc repro`.

## 17. Clean & Reproduce
```bash
rm -f model/help_model*.keras
rm -f metrics/*.json metrics/*.csv
rm -f images/*.png
dvc repro train
```

## 18. Sequence Visualization
`test_train.py` can produce per-sequence prediction plots and (if enabled) attention/importance figures. Disable visual generation with `--no-viz` for batch automation.

## 19. Design Rationale (Brief Reflection)
- Intrinsic attention (instead of post-hoc methods only) provides immediate temporal interpretability and aligns with pedagogical intervention needs.
- Mask-aware attention ensures padded timesteps cannot dilute probability mass.
- Separate attention submodel lowers computational & cognitive overhead for downstream analytics.
- DVC guarantees reproducibility for both data transformations and modeling decisions, crucial in educational settings where traceability is mandated.
- Eliminating inline lambdas future-proofs serialization across TensorFlow versions and mixed deployment targets.

## 20. Suggested Future Enhancements
- ONNX export for broader runtime compatibility.
- Unified explainer combining attention + SHAP per timestep.
- Walk-forward temporal validation to evaluate generalization drift.
- Docker image for hermetic reproducibility.
- Automated fairness auditing (e.g., segment-based performance).

## 21. Quick Commands
```bash
# Full pipeline
dvc repro
# Training only
dvc repro train
# Inference + attention
python src/test_train.py params.yaml model/help_model.keras data/featureselection.csv --output_dir images --test-sequences 10 --no-viz
# Focused heatmap stage (if defined in dvc.yaml)
dvc repro attentionheatmap
# TensorBoard
tensorboard --logdir logs
# Metrics
dvc metrics show
# Plots
dvc plots show
```

## 22. License
MIT License (see LICENSE file).

## 23. Citation
(If publishing, add BibTeX entry here.)

---
If you need deeper troubleshooting guidance or deployment documentation, open an issue or extend the relevant section.
