import os
import argparse
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from train import load, load_time_series
from keras_custom_layers import (
    compute_mask_layer,
    squeeze_last_axis_func,
    mask_attention_scores_func,
    apply_attention_func
)


def load_full_model(model_path: str, mask_value: float) -> tf.keras.Model:
    """Load full trained model with required custom objects for Lambda deserialization."""
    custom_objects = {
        'func': compute_mask_layer(mask_value),
        'squeeze_last_axis_func': squeeze_last_axis_func,
        'mask_attention_scores_func': mask_attention_scores_func,
        'apply_attention_func': apply_attention_func
    }
    return tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)


def build_attention_submodel(full_model: tf.keras.Model) -> tf.keras.Model:
    """Extract attention layer output as standalone submodel; raises if layer missing."""
    try:
        attn_layer = full_model.get_layer('attention_weights')
    except Exception as e:
        raise RuntimeError("Layer 'attention_weights' not found. Was the model trained with use_attention: true?") from e
    return tf.keras.Model(inputs=full_model.input, outputs=attn_layer.output, name='attention_submodel_dynamic')


def load_attention_submodel(model_path: str, mask_value: float) -> tf.keras.Model:
    """Load an attention submodel derived from the main model filename.
    Tries to load <model>_attention.keras first. If not found, loads the full model
    and extracts the layer named 'attention_weights'.
    Adds custom_objects including the closure 'func' created by compute_mask_layer(mask_value).
    """
    attention_path = model_path.replace('.keras', '_attention.keras')
    custom_objects = {
        # Closure returned by compute_mask_layer(mask_value) was serialized under name 'func'
        'func': compute_mask_layer(mask_value),
        'squeeze_last_axis_func': squeeze_last_axis_func,
        'mask_attention_scores_func': mask_attention_scores_func,
        'apply_attention_func': apply_attention_func
    }

    if os.path.exists(attention_path):
        try:
            return tf.keras.models.load_model(attention_path, compile=False, custom_objects=custom_objects)
        except Exception as e:
            print(f"WARNING: Failed to load dedicated attention submodel {attention_path}: {e}. Falling back to full model.")

    # Fallback: load full model and build submodel dynamically
    print("Loading full model to extract attention layer...")
    full_model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    try:
        attn_layer = full_model.get_layer('attention_weights')
    except Exception as e:
        raise RuntimeError("Could not locate layer 'attention_weights' in the full model.") from e
    attention_model = tf.keras.Model(inputs=full_model.input, outputs=attn_layer.output, name='attention_submodel_dynamic')
    return attention_model


def derive_valid_mask(sequence: np.ndarray, mask_value: float) -> np.ndarray:
    """Derive a boolean mask of valid (non-padding) time steps.
    Strategy: a time step is considered padding if ALL feature values equal mask_value.
    sequence shape: (time_steps, features)
    Returns shape: (time_steps,) bool
    """
    # np.isclose to be robust if mask_value is float (even though dataset uses -1)
    return ~np.all(np.isclose(sequence, mask_value), axis=-1)


def renormalize_attention(attention: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Renormalize attention weights only over valid (non-padding) steps.
    Invalid steps are set to np.nan for clearer visualization in heatmap.
    """
    att_copy = attention.astype(float).copy()
    att_copy[~valid_mask] = np.nan
    valid_values = att_copy[valid_mask]
    if valid_values.size > 0:
        att_copy[valid_mask] = valid_values / valid_values.sum()
    return att_copy


def plot_attention_heatmap(attention_weights: np.ndarray,
                           valid_mask: np.ndarray,
                           output_path: str,
                           sequence_labels: np.ndarray | None = None,
                           title: str = "Attention weights over time steps",
                           dpi: int = 300):
    """Create and save a heatmap for a single sequence attention vector.
    attention_weights: vector (time_steps,) possibly containing np.nan for padding.
    valid_mask: boolean mask of real steps.
    sequence_labels: optional array (time_steps, 1 or time_steps,) with target labels per step.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build annotation row (optional) combining label + weight for valid steps
    annotations = []
    if sequence_labels is not None:
        seq_labels_flat = sequence_labels.reshape(-1)
        for i, (w, vm) in enumerate(zip(attention_weights, valid_mask)):
            if not vm or np.isnan(w):
                annotations.append("")
            else:
                annotations.append(f"{w:.2f}\nL:{int(seq_labels_flat[i])}")
    else:
        annotations = [f"{w:.2f}" if (not np.isnan(w) and vm) else "" for w, vm in zip(attention_weights, valid_mask)]

    # Heatmap expects 2D; we provide a single row
    data_2d = np.expand_dims(attention_weights, axis=0)

    plt.figure(figsize=(max(6, len(attention_weights) * 0.25), 1.8))
    sns.heatmap(data_2d,
                cmap="viridis",
                cbar=True,
                xticklabels=list(range(len(attention_weights))),
                yticklabels=["attn"],
                annot=np.array([annotations]),
                fmt="",
                linewidths=0.5,
                linecolor='gray')
    plt.title(title, fontsize=10)
    plt.xlabel("Time step")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def save_attention_csv(attention_weights: np.ndarray, valid_mask: np.ndarray, csv_path: str):
    """Persist per-step attention (with NaNs for padding) to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame({
        'time_step': np.arange(len(attention_weights)),
        'attention_weight': attention_weights,
        'is_valid_step': valid_mask.astype(int)
    })
    df.to_csv(csv_path, index=False)


def generate_model_diagram(model: tf.keras.Model, output_path: str):
    """Generate (and always create) an architecture diagram PNG.
    Tries keras.utils.plot_model; if it fails (e.g., missing pydot/graphviz or custom layer issues),
    creates a placeholder figure so downstream DVC expects file presence.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file=output_path, dpi=200, show_shapes=True, show_layer_names=True)
        print(f"Model diagram saved to {output_path}")
        return
    except Exception as e:
        print(f"WARNING: Could not generate detailed model diagram: {e}. Creating placeholder image.")
    # Fallback placeholder
    plt.figure(figsize=(4, 2))
    plt.text(0.5, 0.5, 'Model diagram not available', ha='center', va='center', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Placeholder model diagram saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LSTM attention heatmap and architecture figure.")
    parser.add_argument('--params-file', required=True, help='Path to params.yaml used in training')
    parser.add_argument('--input-csv-file', required=True, help='CSV file with raw sequential data')
    parser.add_argument('--model-file', required=True, help='Trained Keras model file (.keras)')
    parser.add_argument('--sequence-index', type=int, default=0, help='Index of the test sequence to visualize')
    parser.add_argument('--output-dir', default='images',
                        help='Directory for output figures')
    parser.add_argument('--heatmap-file-name', default='attention_heatmap.png', help='Filename for the heatmap image')
    parser.add_argument('--model-diagram-name', default='model_lstm_attention.png', help='Filename for the model diagram image')
    parser.add_argument('--save-attention-csv', default='metrics/test_attention.csv',
                        help='Optional CSV path to store per-step attention weights')
    parser.add_argument('--top-k', type=int, default=5, help='Print top-K time steps by attention weight')
    args = parser.parse_args()

    # Load params
    with open(args.params_file, 'r') as fd:
        params = yaml.safe_load(fd)

    mask_value = params['model'].get('mask_value', -1)
    percentage_train_size = params['model'].get('percentage_train_size', 70)
    distance_type = params['model'].get('distance_calculation_type', 'artie')

    # 1. Load full model first to know expected feature dimension
    full_model = load_full_model(args.model_file, mask_value)
    expected_feat_dim = full_model.input_shape[-1]
    if expected_feat_dim is None:
        raise RuntimeError("Could not infer feature dimension from model input shape.")
    print(f"Model expects {expected_feat_dim} features per time step.")

    # 2. Load raw data and build sequences with current preprocessing
    df_raw, max_time_steps, columns = load(args.input_csv_file)
    _, train_x, train_y, test_x, test_y = load_time_series(df_raw, max_time_steps, columns, mask_value,
                                                           percentage_train_size, distance_type)
    if test_x is None or test_x.shape[0] == 0:
        raise RuntimeError("Empty test set.")

    current_feat_dim = test_x.shape[2]
    if current_feat_dim > expected_feat_dim:
        print(f"WARNING: dataset has {current_feat_dim} features; model trained with {expected_feat_dim}. Truncating extra {current_feat_dim - expected_feat_dim} columns (assuming they were appended).")
        test_x = test_x[:, :, :expected_feat_dim]
        train_x = train_x[:, :, :expected_feat_dim]
    elif current_feat_dim < expected_feat_dim:
        raise RuntimeError(f"Dataset has {current_feat_dim} features but model expects {expected_feat_dim}. Cannot proceed. Recreate model or align preprocessing.")

    # 3. Select sequence
    if args.sequence_index < 0 or args.sequence_index >= test_x.shape[0]:
        raise IndexError(f"sequence_index {args.sequence_index} out of range [0, {test_x.shape[0]-1}].")
    sequence_x = test_x[args.sequence_index]
    sequence_y = test_y[args.sequence_index]

    # 4. Build / load attention submodel
    attention_model = build_attention_submodel(full_model)

    # 5. Predict attention weights
    attention_prediction = attention_model.predict(sequence_x[None, ...], verbose=0)
    if attention_prediction.ndim != 2 or attention_prediction.shape[0] != 1:
        raise ValueError(f"Unexpected attention shape {attention_prediction.shape}; expected (1, T).")
    raw_attention = attention_prediction[0]

    # 6. Mask & renormalize
    valid_mask = derive_valid_mask(sequence_x, mask_value)
    renorm_attention = renormalize_attention(raw_attention, valid_mask)

    # 7. Persist CSV
    if args.save_attention_csv:
        save_attention_csv(renorm_attention, valid_mask, args.save_attention_csv)
        print(f"CSV guardado: {args.save_attention_csv}")

    # 8. Top-K
    if args.top_k > 0:
        valid_indices = np.where(valid_mask)[0]
        valid_values = renorm_attention[valid_mask]
        if valid_values.size:
            top_order = np.argsort(valid_values)[-min(args.top_k, valid_values.size):][::-1]
            print("Top pasos por atención:")
            for r, idx in enumerate(top_order, 1):
                timestep = valid_indices[idx]
                print(f"  {r}. paso={timestep} peso={valid_values[idx]:.4f} label={int(sequence_y[timestep][0])}")
        else:
            print("Sin pasos válidos (todo padding).")

    # 9. Heatmap
    heatmap_path = os.path.join(args.output_dir, args.heatmap_file_name)
    plot_attention_heatmap(renorm_attention, valid_mask, heatmap_path, sequence_labels=sequence_y,
                           title=f"Attention weights (sequence {args.sequence_index})")
    print(f"Heatmap guardado en {heatmap_path}")

    # 10. Diagrama modelo
    model_diagram_path = os.path.join(args.output_dir, args.model_diagram_name)
    generate_model_diagram(full_model, model_diagram_path)

    print("Done.")


if __name__ == '__main__':
    main()
