import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import yaml
import sys

# Lista exacta de características esperadas por el modelo (15)
EXPECTED_FEATURES = [
    "student_sex",
    "student_mother_tongue",
    "student_age",
    "student_competence",
    "exercise_skill_parallelism",
    "exercise_skill_logical_thinking",
    "exercise_skill_flow_control",
    "exercise_skill_user_interactivity",
    "exercise_skill_information_representation",
    "exercise_skill_abstraction",
    "exercise_skill_synchronization",
    "exercise_level",
    "solution_distance_total_distance",
    "seconds_help_open",
    "total_seconds",
]

def configure_gpu(use_gpu=False):
    """Configure GPU usage for TensorFlow"""
    print(f"\n{'Enabling' if use_gpu else 'Disabling'} GPU acceleration...")

    # List available physical devices
    print("Available physical devices:")
    print(tf.config.list_physical_devices())

    if use_gpu:
        # Try to configure GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Available GPUs: {gpus}")
            try:
                # Specific configuration for Metal GPU on Mac
                tf.config.set_visible_devices(gpus, 'GPU')

                # Optimal configuration for Metal
                # Limit memory usage to avoid OOM issues
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                )

                # Configuration for better performance with Metal (if available)
                try:
                    tf.config.optimizer.set_jit(False)  # Disable XLA which may cause issues with Metal
                except Exception as e:
                    print(f"Warning when configuring optimizer: {e}")

                print("Metal GPU enabled for inference with safe settings")
            except RuntimeError as e:
                print(f"Error configuring GPU: {e}")
                print("Continuing with CPU due to errors in GPU configuration")
                use_gpu = False
                tf.config.set_visible_devices([], 'GPU')  # Disable GPU
        else:
            print("No GPUs found. Using CPU.")
            use_gpu = False
    else:
        # Disable GPU usage
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled. Using CPU for inference.")

    # Verify active device for operations
    print(f"Device that will be used: {'GPU' if use_gpu else 'CPU'}")
    return use_gpu


def load_test_data(csv_path, mask_value=-1, max_sequences=10):
    """
    Load and preprocess test data from a CSV file.
    Limited to a maximum number of sequences.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Exclude target column if it exists
    exclude_columns = ['request_help'] if 'request_help' in df.columns else []
    y_true = None
    if 'request_help' in df.columns:
        y_true = df['request_help'].values

    # Asegurar que existen todas las columnas esperadas; rellenar faltantes con 0.0
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        print(f"WARNING: faltan columnas en el CSV necesarias para el modelo: {missing}. Se rellenarán con 0.0")
        for c in missing:
            df[c] = 0.0

    # Seleccionar exactamente las columnas esperadas y en el orden correcto
    features = df[EXPECTED_FEATURES].copy()
    num_columns = features.shape[1]

    # If there's no 'total_seconds' column, treat as a single sequence
    if 'total_seconds' not in df.columns:
        print("No 'total_seconds' column found. Treating data as a single sequence.")
        X = features.values
        return np.array([X]), y_true, df

    # Process sequences based on 'total_seconds'
    print("Processing time sequences...")
    sequences = []
    last_time = -1
    current_sequence = []
    max_length = 0
    sequence_count = 0

    # Group by sequences
    for i, row in enumerate(features.values):
        current_time = df['total_seconds'].iloc[i]

        # If last_time > current_time, it's a new sequence
        if last_time > current_time and len(current_sequence) > 0:
            sequences.append(np.array(current_sequence))
            max_length = max(max_length, len(current_sequence))
            current_sequence = []
            sequence_count += 1

            # Limit to max_sequences
            if sequence_count >= max_sequences:
                break

        current_sequence.append(row)
        last_time = current_time

    # Add the last sequence if we haven't reached the limit
    if current_sequence and sequence_count < max_sequences:
        sequences.append(np.array(current_sequence))
        max_length = max(max_length, len(current_sequence))

    print(f"Found {len(sequences)} sequences (limited to {max_sequences}), maximum length: {max_length}")

    # Apply padding to sequences
    padded_sequences = []
    for seq in sequences:
        padding_size = max_length - seq.shape[0]
        if padding_size > 0:
            padding = np.full((padding_size, num_columns), mask_value)
            seq_padded = np.vstack([seq, padding])
            padded_sequences.append(seq_padded)
        else:
            padded_sequences.append(seq)

    X_padded = np.array(padded_sequences)
    print(f"Final data shape: {X_padded.shape}")

    return X_padded, y_true, df


def analyze_predictions(model, attention_model, X, feature_names, mask_value=-1, threshold=0.5):
    """
    Analyze and explain model predictions.
    """
    print("\nAnalyzing model predictions...")

    # Make predictions
    predictions = model.predict(X)

    # Initialize results
    results = []

    # For each sequence
    for i, sequence in enumerate(X):
        seq_predictions = predictions[i].flatten()
        # Filter masked values
        valid_indices = [j for j, row in enumerate(sequence) if not np.all(row == mask_value)]
        valid_preds = seq_predictions[valid_indices]

        # Count positive predictions
        num_positives = np.sum(valid_preds >= threshold)
        total = len(valid_preds)

        # Calculate percentage of help requests
        help_percentage = 0 if total == 0 else (num_positives / total) * 100

        print(f"\nSequence {i+1}:")
        print(f"  - Length: {len(valid_indices)} time steps")
        print(f"  - Positive predictions: {num_positives}/{total} ({help_percentage:.1f}%)")

        # If there are positive predictions, show key moments
        if num_positives > 0:
            print("  - Key moments where help is predicted:")
            for j in valid_indices:
                if seq_predictions[j] >= threshold:
                    print(f"    * Step {j+1}: {seq_predictions[j]:.4f}")

        # Calculate attention weights if there's an attention model
        feature_importance = None
        if attention_model is not None:
            attention_weights = attention_model.predict(X[i:i+1])[0]
            # Only consider weights for valid steps
            valid_weights = attention_weights[valid_indices]

            if len(valid_weights) > 0:
                print("  - Attention analysis:")
                # Find moments with highest attention
                important_indices = np.argsort(valid_weights)[-3:]
                for idx in reversed(important_indices):
                    current_idx = valid_indices[idx]
                    print(f"    * Step {current_idx+1}: Attention weight {valid_weights[idx]:.4f}")

                # Calculate feature importance based on sequence values
                valid_sequence = sequence[valid_indices]
                feature_importance = np.zeros(valid_sequence.shape[1])
                for j, weight in enumerate(valid_weights):
                    feature_importance += weight * np.abs(valid_sequence[j])

                # Normalize importance
                if np.sum(feature_importance) > 0:
                    feature_importance = feature_importance / np.sum(feature_importance)

                    # Show most important features
                    print("  - Most influential features:")
                    sorted_indices = np.argsort(feature_importance)[-5:]
                    for idx in reversed(sorted_indices):
                        if idx < len(feature_names):
                            print(f"    * {feature_names[idx]}: {feature_importance[idx]:.4f}")

        # Save results
        results.append({
            'sequence_id': i+1,
            'length': len(valid_indices),
            'num_positives': num_positives,
            'total': total,
            'help_percentage': help_percentage,
            'feature_importance': feature_importance
        })

    return results, predictions


def visualize_predictions(X, predictions, results, df, output_dir="visualizations"):
    """
    Generate visualizations for predictions and feature importance.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nGenerating visualizations in {output_dir}...")

    # For each sequence
    for i, result in enumerate(results):
        if result['length'] == 0:
            continue

        # Create figure for predictions
        plt.figure(figsize=(12, 6))

        # Get valid predictions
        sequence = X[i]
        preds = predictions[i].flatten()
        valid_indices = [j for j, row in enumerate(sequence) if not np.all(row == -1)]
        valid_preds = preds[valid_indices]

        # Plot predictions
        plt.plot(valid_preds, marker='o', linestyle='-', label='Help request probability')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')

        # Configure plot
        plt.title(f'Predictions for Sequence {i+1}')
        plt.xlabel('Time step')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save plot
        plt.savefig(f"{output_dir}/sequence_{i+1}_predictions.png")
        plt.close()

        # If there's feature importance information
        if result['feature_importance'] is not None and np.sum(result['feature_importance']) > 0:
            # Create bar chart for feature importance
            plt.figure(figsize=(12, 8))

            # Get feature names and values
            feature_names = df.columns.drop(['request_help', 'total_seconds'], errors='ignore').tolist()
            importance = result['feature_importance']

            # Show only the top 10 most important features
            if len(feature_names) > 10:
                # Asegurarse de que top_indices no contenga índices fuera de rango
                top_indices = [i for i in np.argsort(importance)[-10:] if i < len(feature_names)]
                top_features = [feature_names[i] for i in top_indices]
                top_importance = importance[top_indices]

                # Comprobar que hay características para mostrar
                if len(top_features) > 0:
                    # Sort for visualization
                    sorted_indices = np.argsort(top_importance)
                    # Proteger contra índices fuera de rango
                    sorted_features = []
                    sorted_importance = []
                    for idx in sorted_indices:
                        if idx < len(top_features):
                            sorted_features.append(top_features[idx])
                            sorted_importance.append(top_importance[idx])

                    # Verificar que hay datos para graficar
                    if len(sorted_features) > 0:
                        plt.barh(sorted_features, sorted_importance)
                    else:
                        print(f"  - No hay suficientes características con importancia para la secuencia {i+1}")
                else:
                    print(f"  - No hay características con importancia para la secuencia {i+1}")
            else:
                # Sort for visualization
                sorted_indices = np.argsort(importance)
                # Filtrar índices fuera de rango
                valid_indices = [i for i in sorted_indices if i < len(feature_names)]
                sorted_features = [feature_names[i] for i in valid_indices]
                sorted_importance = importance[valid_indices]

                # Verificar que hay datos para graficar
                if len(sorted_features) > 0:
                    plt.barh(sorted_features, sorted_importance)
                else:
                    print(f"  - No hay características con importancia para la secuencia {i+1}")

            plt.title(f'Feature Importance for Sequence {i+1}')
            plt.xlabel('Relative Importance')
            plt.tight_layout()

            # Save plot
            plt.savefig(f"{output_dir}/sequence_{i+1}_feature_importance.png")
            plt.close()

    print(f"Visualizations saved in {output_dir}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test and explain keras model predictions')
    parser.add_argument('params_file', help='YAML file with model parameters')
    parser.add_argument('model_path', help='Path to the Keras model file')
    parser.add_argument('data_file', help='CSV file with data to evaluate')
    parser.add_argument('--output_dir', default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU for inference')
    parser.add_argument('--test-sequences', type=int, default=10, help='Maximum number of sequences to process')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization generation')

    args = parser.parse_args()

    # Check file existence
    if not os.path.exists(args.params_file):
        print(f"ERROR: Parameters file '{args.params_file}' not found.")
        return 1

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file '{args.model_path}' not found.")
        return 1

    if not os.path.exists(args.data_file):
        print(f"ERROR: Data file '{args.data_file}' not found.")
        return 1

    # Configure GPU usage
    configure_gpu(args.use_gpu)

    # Load parameters
    with open(args.params_file, 'r') as f:
        params = yaml.safe_load(f)

    mask_value = params['model'].get('mask_value', -1)
    use_attention = params['model'].get('use_attention', False)

    # Load main model (sin objetos personalizados)
    print(f"\nLoading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path, compile=False, safe_mode=False)
        model.summary()
    except Exception as e:
        print(f"FINAL ERROR: Could not load the model: {e}")
        return 1

    # Load attention model if it exists (opcional)
    attention_model = None
    attention_path = args.model_path.replace('.keras', '_attention.keras')
    if os.path.exists(attention_path) and use_attention:
        try:
            print(f"\nLoading attention model from {attention_path}...")
            attention_model = tf.keras.models.load_model(attention_path, compile=False, safe_mode=False)
            attention_model.summary()
        except Exception as e:
            print(f"ERROR: Could not load attention model: {e}")
            print("Continuing without attention model...")

    # Load test data (limited to specified number of sequences)
    X, y_true, df = load_test_data(args.data_file, mask_value, max_sequences=args.test_sequences)

    # Usar los nombres de features en el orden esperado
    feature_names = EXPECTED_FEATURES

    # Explicar y obtener predicciones
    results, predictions = analyze_predictions(model, attention_model, X, feature_names, mask_value)

    # Save predictions to CSV for DVC
    print("\nSaving predictions to CSV files...")

    # Create metrics directory if it doesn't exist
    os.makedirs('metrics', exist_ok=True)

    # Save predictions
    predictions_data = []
    for i, pred_seq in enumerate(predictions):
        for j, pred in enumerate(pred_seq.flatten()):
            predictions_data.append({
                'sequence_id': i + 1,
                'time_step': j + 1,
                'prediction': pred
            })

    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv('metrics/test_predictions.csv', index=False)
    print("Predictions saved to metrics/test_predictions.csv")

    # Save attention weights if available
    if attention_model is not None:
        print("Extracting attention weights...")
        attention_data = []
        for i in range(len(X)):
            try:
                attention_weights = attention_model.predict(X[i:i+1], verbose=0)[0]
                for j, weight in enumerate(attention_weights.flatten()):
                    attention_data.append({
                        'sequence_id': i + 1,
                        'time_step': j + 1,
                        'attention_weight': weight
                    })
            except Exception as e:
                print(f"Error getting attention weights for sequence {i+1}: {e}")

        if attention_data:
            attention_df = pd.DataFrame(attention_data)
            attention_df.to_csv('metrics/test_attention.csv', index=False)
            print("Attention weights saved to metrics/test_attention.csv")
        else:
            # Create empty attention file if there was an error
            empty_attention_df = pd.DataFrame(columns=['sequence_id', 'time_step', 'attention_weight'])
            empty_attention_df.to_csv('metrics/test_attention.csv', index=False)
            print("Empty attention file created due to errors")
    else:
        # Create empty attention file if no attention model
        empty_attention_df = pd.DataFrame(columns=['sequence_id', 'time_step', 'attention_weight'])
        empty_attention_df.to_csv('metrics/test_attention.csv', index=False)
        print("Empty attention file created (no attention model available)")

    # Visualize predictions (si no está deshabilitado)
    if not args.no_viz:
        visualize_predictions(X, predictions, results, df, args.output_dir)
    else:
        print("\nVisualizaciones deshabilitadas por el parámetro --no-viz")

    print("\nAnalysis completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())