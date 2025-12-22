import warnings
warnings.filterwarnings("ignore", message="Layer 'lambda.*' .* does not support masking.*")

import pandas as pd
import numpy as np
import math
import os
import yaml
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional
import argparse
from sklearn.metrics import precision_recall_curve
import json

# Set seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)
np.random.seed(1)

# Function to load the csv file
def load(file):
    # Reads the csv file (robust to different separators)
    try:
        df = pd.read_csv(file)
    except Exception:
        try:
            df = pd.read_csv(file, sep=None, engine='python')  # auto-detect separator
        except Exception:
            df = pd.read_csv(file, sep=';')  # final fallback for semicolon-separated files
    columns = len(df.columns) - 1

    # Read max_time_steps from precomputed analysis
    max_time_steps = 1
    try:
        ts_path = os.path.join('data', 'time_steps_analysis.csv')
        if os.path.exists(ts_path):
            ts_df = pd.read_csv(ts_path)
            if 'time_steps' in ts_df.columns and not ts_df.empty:
                max_time_steps = int(ts_df['time_steps'].max())
    except Exception as e:
        print(f"Warning: could not read time steps analysis file: {e}. Falling back to default max_time_steps=1")

    return df, max_time_steps, columns


# Function to create the padding and the masking
def padding_masking(time_series_x, time_series_y, max_time_steps, columns, mask_value):
    diff_number_steps = 0

    # We compare the current number of steps with max_time_steps
    if time_series_x.shape[0] < max_time_steps:
        diff_number_steps = (max_time_steps - time_series_x.shape[0])

    # We create a new array to mask
    diff_array_x = np.full((diff_number_steps, columns), mask_value)
    diff_array_y = np.full((diff_number_steps, 1), mask_value)

    time_series_x = np.vstack([time_series_x, diff_array_x])
    time_series_y = np.vstack([time_series_y, diff_array_y])

    return time_series_x, time_series_y


# Function to load the time series and separates it into features and class
def load_time_series(df, max_time_steps, columns, mask_value, percentage_train, distance_calculation_type):
    # Filters by students with age less than or equal to 15
    df = df.dropna(subset=['student_age'])
    df = df[df['student_age'] <= 15]
    # Ensure student_age is numeric
    df = df[pd.to_numeric(df['student_age'], errors='coerce').notna()]
    # Reset index after filtering to ensure proper alignment
    df = df.reset_index(drop=True)

    # Determine columns to drop from features (label + distance set + grouping columns)
    apted_columns = ["group_id","date_time","request_help", "apted_distance", "tree_grade"]
    artie_columns = ["group_id","date_time","request_help", "solution_distance_family_distance",
                     "solution_distance_element_distance", "solution_distance_position_distance",
                     "solution_distance_input_distance", "bed_distance", "grade"]

    if distance_calculation_type.lower() == 'artie':
        base_drop = [col for col in apted_columns if col in df.columns]
    else:
        base_drop = [col for col in artie_columns if col in df.columns]

    # Always drop grouping columns from features
    for col in ['group_id', 'date_time']:
        if col not in base_drop and col in df.columns:
            base_drop.append(col)

    # If grouping columns exist, group by group_id + date (yyyy-mm-dd)
    if 'group_id' in df.columns and 'date_time' in df.columns:
        df_work = df.copy()
        # Build date column
        df_work['__date'] = pd.to_datetime(df_work['date_time'], errors='coerce').dt.strftime('%Y-%m-%d')
        df_work = df_work.dropna(subset=['__date'])

        # Sort for deterministic ordering within sequences
        if 'total_seconds' in df_work.columns:
            df_work = df_work.sort_values(by=['group_id', '__date', 'total_seconds']).reset_index(drop=True)
        else:
            df_work = df_work.sort_values(by=['group_id', '__date']).reset_index(drop=True)

        sample_x = None
        sample_y = None

        # Iterate groups and build sequences
        for (_, _), gdf in df_work.groupby(['group_id', '__date'], sort=False):
            # Prepare features X and labels y for this group
            df_y = gdf["request_help"].to_numpy().reshape(-1, 1)

            # Drop columns not used for features
            cols_to_drop = [c for c in base_drop if c in gdf.columns]
            if '__date' in gdf.columns:
                cols_to_drop.append('__date')
            df_X = gdf.drop(columns=cols_to_drop, axis=1, errors='ignore')

            # Keep ordering as in gdf (already sorted)
            X = df_X.to_numpy()
            y = df_y

            # Update feature dimension
            columns = df_X.shape[1]

            # Truncate or pad to max_time_steps
            if X.shape[0] > max_time_steps:
                X = X[:max_time_steps, :]
                y = y[:max_time_steps, :]
            elif X.shape[0] < max_time_steps:
                X, y = padding_masking(X, y, max_time_steps, columns, mask_value)

            # Accumulate samples
            X = np.array([X])
            y = np.array([y])
            if sample_x is None:
                sample_x = X
                sample_y = y
            else:
                sample_x = np.vstack([sample_x, X])
                sample_y = np.vstack([sample_y, y])

        # Compute train/test split
        train_size = math.floor(sample_x.shape[0] * percentage_train / 100)
        train_x = sample_x[:train_size]
        train_y = sample_y[:train_size]
        test_x = sample_x[train_size + 1:]
        test_y = sample_y[train_size + 1:]

        return df, train_x, train_y, test_x, test_y

    # Fallback: if grouping columns are missing, use previous time reset logic
    df_X = df.drop(axis=1, columns=[c for c in base_drop if c in df.columns])
    columns = df_X.shape[1]
    df_y = df["request_help"]
    last_step_seconds = -1
    time_steps_x = None
    time_steps_y = None
    sample_x = None
    sample_y = None
    for index, row in df_X.iterrows():
        current_seconds = df_X["total_seconds"][index] if "total_seconds" in df_X.columns else index
        if time_steps_x is None:
            time_steps_x = np.array([row])
            time_steps_y = np.array([df_y.iloc[[index]]])
        else:
            time_steps_x = np.vstack([time_steps_x, row])
            time_steps_y = np.vstack([time_steps_y, df_y.iloc[[index]]])
        if last_step_seconds > current_seconds:
            time_steps_x, time_steps_y = padding_masking(time_steps_x, time_steps_y, max_time_steps, columns, mask_value)
            if sample_x is None:
                sample_x = np.array([time_steps_x])
                sample_y = np.array([time_steps_y])
            else:
                sample_x = np.vstack([sample_x, [time_steps_x]])
                sample_y = np.vstack([sample_y, [time_steps_y]])
            time_steps_x = None
            time_steps_y = None
        last_step_seconds = current_seconds

    train_size = math.floor(sample_x.shape[0] * percentage_train / 100)
    train_x = sample_x[:train_size]
    train_y = sample_y[:train_size]
    test_x = sample_x[train_size + 1:]
    test_y = sample_y[train_size + 1:]
    return df, train_x, train_y, test_x, test_y


# Function to generate the model
def generate_model(shape, mask_value, lstm_units, return_sequences=False, second_lstm_layer=False, use_dropout=False,
                   dropout_value=0.5, use_bidirectional=True, use_attention=False):
    inputs = tf.keras.Input(shape=shape)
    masked = layers.Masking(mask_value=mask_value)(inputs)

    # Optional first LSTM stack
    if second_lstm_layer:
        if use_bidirectional:
            x = Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(masked)
        else:
            x = layers.LSTM(lstm_units, return_sequences=True)(masked)
        if use_dropout:
            x = layers.Dropout(dropout_value)(x)
    else:
        x = masked

    # Main LSTM layer
    if use_bidirectional:
        x = Bidirectional(layers.LSTM(lstm_units, return_sequences=return_sequences))(x)
    else:
        x = layers.LSTM(lstm_units, return_sequences=return_sequences)(x)

    if use_dropout:
        x = layers.Dropout(dropout_value)(x)

    # Atención opcional sin Lambda
    attention_weights = None
    if use_attention and return_sequences:
        attn_scores = layers.Dense(1, activation='tanh', name='attention_score')(x)          # (B, T, 1)
        attn_scores_2d = layers.Reshape((-1,), name='attention_scores_2d')(attn_scores)      # (B, T)
        attention_weights = layers.Softmax(axis=1, name='attention_weights')(attn_scores_2d) # (B, T)
        attn_weights_exp = layers.Reshape((-1, 1), name='attention_weights_exp')(attention_weights)
        x = layers.Multiply(name='apply_attention')([x, attn_weights_exp])

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Exponer submodelo de atención si existe
    if use_attention and attention_weights is not None:
        attention_model = tf.keras.Model(inputs=inputs, outputs=attention_weights, name='attention_submodel')
        model.attention_model = attention_model

    return model

# Function to compute sample weights for time series data
def compute_sample_weights_for_time_series(train_y, mask_value, class_weights):
    """
    Compute sample weights for each time step in the time series data
    Optimized version using vectorized operations
    """
    # Initialize weights with ones
    sample_weights = np.ones_like(train_y, dtype=float)

    # Convert train_y to a numpy array if not already
    if not isinstance(train_y, np.ndarray):
        train_y = np.array(train_y)

    # Create mask for masked values
    mask = (train_y == mask_value)

    # Assign weight 0 to masked values
    sample_weights[mask] = 0.0

    # For each class, assign the corresponding weight
    for class_label, weight in class_weights.items():
        # Ensure class_label is a numeric value
        class_label_value = float(class_label)
        # Create mask for this class (values equal to class_label and not masked)
        class_mask = (train_y == class_label_value) & ~mask
        # Assign weight
        sample_weights[class_mask] = weight

    return sample_weights


if __name__ == "__main__":
    print(tf.__version__)

    parser = argparse.ArgumentParser(description="Train HelpModel with configurable arguments")
    parser.add_argument('--params-file', required=True, help='Path to params.yaml')
    parser.add_argument('--input-csv-file', required=True, help='Input CSV file')
    parser.add_argument('--output-model-file', required=True, help='Output model file (.keras)')
    parser.add_argument('--plots-file-name', required=True, help='CSV file for training plots')
    parser.add_argument('--metrics-file-name', required=True, help='JSON file for metrics')
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU usage')
    parser.add_argument('--output-dir', default='images', help='Directory for output images')
    parser.add_argument('--log-dir', default='logs', help='TensorBoard log directory')
    args = parser.parse_args()

    params_file = args.params_file
    input_csv_file = args.input_csv_file
    output_model_file = args.output_model_file
    plots_file_name = args.plots_file_name
    metrics_file_name = args.metrics_file_name
    use_gpu = args.use_gpu
    output_dir = args.output_dir
    log_dir = args.log_dir

    os.makedirs(os.path.dirname(plots_file_name), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_file_name), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(params_file, 'r') as fd:
        params = yaml.safe_load(fd)

    distance_calculation_type = params['model'].get('distance_calculation_type', 'artie')
    mask_value = params['model'].get('mask_value', -1)
    percentage_train_size = params['model'].get('percentage_train_size', 70)
    initial_learning_rate = params['model'].get('initial_learning_rate', 0.001)
    lstm_units = params['model'].get('lstm_units', 256)
    return_sequences = params['model'].get('return_sequences', True)
    second_lstm_layer = params['model'].get('second_lstm_layer', True)
    use_dropouts = params['model'].get('use_dropouts', True)
    dropout_value = params['model'].get('dropout_value', 0.5)
    use_bidirectional = params['model'].get('use_bidirectional', True)
    use_attention = params['model'].get('use_attention', False)
    training_epochs = params['model'].get('training_epochs', 100)
    training_batch_size = params['model'].get('training_batch_size', 32)
    training_class_weights = params['model'].get('training_class_weights', True)
    training_early_stopping_patience = params['model'].get('training_early_stopping_patience', 10)
    training_reduce_lr_patience = params['model'].get('training_reduce_lr_patience', 5)
    training_reduce_lr_factor = params['model'].get('training_reduce_lr_factor', 0.1)
    show_summary = params['model'].get('show_summary', True)

    print(tf.config.list_physical_devices())

    if use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Available GPUs: {gpus}")
            try:
                tf.config.set_visible_devices(gpus, 'GPU')
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                )
                try:
                    tf.config.optimizer.set_jit(False)
                except Exception:
                    pass
                print("Metal GPU enabled for training with safe settings")
            except RuntimeError as e:
                print(f"Error configuring GPU: {e}")
                use_gpu = False
        else:
            print("No GPUs found. Using CPU for training.")
            use_gpu = False
    else:
        try:
            tf.config.set_visible_devices([], 'GPU')
            print("GPU disabled for training. Using CPU.")
        except Exception as e:
            print(f"Could not disable GPU explicitly: {e}")

    df, max_time_steps, columns = load(input_csv_file)

    df, train_x, train_y, test_x, test_y = load_time_series(
        df, max_time_steps, columns, mask_value, percentage_train_size, distance_calculation_type
    )

    shape = (None, train_x.shape[2])
    model = generate_model(
        shape, mask_value, lstm_units, return_sequences, second_lstm_layer,
        use_dropouts, dropout_value, use_bidirectional, use_attention
    )

    print("Device configuration for training:")
    print("- Visible devices:", tf.config.get_visible_devices())
    print("- Using GPU:", use_gpu)

    class LearningRateLogger(tf.keras.callbacks.Callback):
        def __init__(self, log_dir):
            super().__init__()
            self.log_dir = log_dir
            self.lr_values = []
            self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'learning_rate'))
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            try:
                opt = self.model.optimizer
                lr_t = opt.learning_rate
                if isinstance(lr_t, tf.keras.optimizers.schedules.LearningRateSchedule):
                    lr_v = lr_t(opt.iterations)
                else:
                    lr_v = lr_t
                if hasattr(lr_v, 'numpy'):
                    lr_v = lr_v.numpy()
            except Exception:
                lr_v = None
            self.lr_values.append(lr_v)
            if lr_v is not None:
                with self.writer.as_default():
                    tf.summary.scalar('learning_rate', data=lr_v, step=epoch)
            print(f"\nLearning rate at epoch {epoch+1}: {lr_v}")

    class ValidationMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_data, mask_value, log_path, log_dir=None):
            super().__init__()
            self.val_x, self.val_y = val_data
            self.mask_value = mask_value
            self.log_path = log_path
            self.best_f1 = -1
            self.best_threshold = 0.5
            self.history = []
            self.writer = None
            if log_dir is not None:
                self.writer = tf.summary.create_file_writer(os.path.join(log_dir, 'validation_custom'))
        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(self.val_x, verbose=0)
            y_true = self.val_y.reshape(-1)
            y_pred = preds.reshape(-1)
            mask = (y_true != self.mask_value)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
            idx = f1_scores.argmax()
            best_thr = thresholds[idx] if idx < len(thresholds) else 0.5
            best_f1 = f1_scores[idx]
            if best_f1 > self.best_f1:
                self.best_f1 = best_f1
                self.best_threshold = best_thr
            self.history.append({
                "epoch": epoch + 1,
                "f1": float(best_f1),
                "best_f1_so_far": float(self.best_f1),
                "threshold_epoch": float(best_thr),
                "best_threshold": float(self.best_threshold)
            })
            if self.writer is not None:
                with self.writer.as_default():
                    tf.summary.scalar('val_f1', best_f1, step=epoch)
                    tf.summary.scalar('val_best_threshold', self.best_threshold, step=epoch)
            print(f"[ValMetrics] Epoch {epoch+1}: F1={best_f1:.4f} thr={best_thr:.3f} (best_f1={self.best_f1:.4f} best_thr={self.best_threshold:.3f})")
        def on_train_end(self, logs=None):
            try:
                with open(self.log_path, "w") as f:
                    json.dump({
                        "best_f1": float(self.best_f1),
                        "best_threshold": float(self.best_threshold),
                        "epochs": self.history
                    }, f, indent=2)
                if self.writer is not None:
                    with self.writer.as_default():
                        tf.summary.scalar('final_best_f1', self.best_f1, step=0)
                        tf.summary.scalar('final_best_threshold', self.best_threshold, step=0)
                print(f"Validation threshold metrics saved to {self.log_path}")
            except Exception as e:
                print(f"Could not save validation threshold metrics: {e}")

    steps_per_epoch = max(1, math.ceil(train_x.shape[0] / training_batch_size))
    decay_steps = steps_per_epoch * training_epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_learning_rate,
                                                            decay_steps=decay_steps, alpha=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics=['binary_accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc', curve='PR')]
    )

    lr_logger = LearningRateLogger(log_dir)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True,
                                                    write_images=False, update_freq='epoch')

    val_metrics_cb = ValidationMetricsCallback((test_x, test_y), mask_value,
                                               log_path=os.path.join(os.path.dirname(metrics_file_name), 'val_threshold_metrics.json'),
                                               log_dir=log_dir)

    callbacks = [lr_logger, tensorboard_cb,
                 tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=max(15, training_early_stopping_patience),
                                                  mode='max', restore_best_weights=True),
                 val_metrics_cb]

    if training_class_weights:
        train_y_flat = train_y.reshape(-1)
        train_y_flat = train_y_flat[train_y_flat != mask_value]
        classes = np.unique(train_y_flat)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y_flat)
        class_weights = dict(zip(classes, weights))
        mean_w = np.mean(list(class_weights.values()))
        for k in class_weights:
            class_weights[k] = float(min(class_weights[k] / mean_w, 8.0))
        print("Normalized class weights:", class_weights)
        sample_weights = compute_sample_weights_for_time_series(train_y, mask_value, class_weights)
    else:
        sample_weights = None

    effective_batch = training_batch_size
    print(f"Effective batch size: {effective_batch}")
    print("Configured metrics:", [m.name if hasattr(m, 'name') else m for m in model.metrics])

    history = model.fit(train_x, train_y,
                        epochs=training_epochs,
                        batch_size=effective_batch,
                        validation_data=(test_x, test_y),
                        verbose=1,
                        shuffle=False,
                        callbacks=callbacks,
                        sample_weight=sample_weights)

    if hasattr(val_metrics_cb, 'history') and val_metrics_cb.history:
        f1_list = [e['f1'] for e in val_metrics_cb.history]
        best_f1_list = [e['best_f1_so_far'] for e in val_metrics_cb.history]
        best_thr_list = [e['best_threshold'] for e in val_metrics_cb.history]
        history.history['f1'] = f1_list
        history.history['best_f1_so_far'] = best_f1_list
        history.history['best_threshold'] = best_thr_list

    print("\nTraining completed successfully. Saving model...")
    try:
        model.save(output_model_file, save_format='keras')
        print(f"Model saved to {output_model_file} in modern format")
    except Exception as e:
        print(f"Error saving in modern format: {e}")
        model.save(output_model_file)
        print(f"Model saved to {output_model_file} in legacy HDF5 format")

    if hasattr(model, 'attention_model'):
        attention_model_path = output_model_file.replace('.keras', '_attention.keras')
        try:
            model.attention_model.save(attention_model_path, save_format='keras')
        except Exception:
            model.attention_model.save(attention_model_path)
        print(f"Attention submodel saved to {attention_model_path}")

    if not history.history:
        print("WARNING: The training history is empty.")
    else:
        if len(lr_logger.lr_values) == len(history.history['loss']):
            history.history['lr'] = lr_logger.lr_values
        else:
            history.history['lr'] = lr_logger.lr_values[:len(history.history['loss'])]
        hist_df = pd.DataFrame(history.history)
        print(f"Model trained during {len(hist_df['loss'])} epochs.")
        if 'precision' in hist_df and 'recall' in hist_df:
            print(f"Final metrics: Precision: {hist_df['precision'].iloc[-1]:.4f}, Recall: {hist_df['recall'].iloc[-1]:.4f}")
        else:
            print("Available metrics:", list(hist_df.columns))
            print(f"Final binary_accuracy: {hist_df['binary_accuracy'].iloc[-1]:.4f}")

    if show_summary:
        print("\nModel summary:")
        model.summary()
        try:
            tf.keras.utils.plot_model(model, to_file=os.path.join(output_dir, 'model.png'), dpi=200)
            print(f"Model diagram saved to {os.path.join(output_dir, 'model.png')}")
        except Exception as e:
            print(f"Error generating model diagram: {e}")

    if history.history:
        try:
            hist_df.to_csv(plots_file_name, index_label='epoch')
        except Exception as e:
            print(f"Could not write plots CSV {plots_file_name}: {e}")
        metrics_data = {'loss': hist_df['loss'].mean(), 'binary_accuracy': hist_df['binary_accuracy'].mean()}
        if 'precision' in hist_df:
            metrics_data['precision'] = hist_df['precision'].mean()
        if 'recall' in hist_df:
            metrics_data['recall'] = hist_df['recall'].mean()
        if 'auc' in hist_df:
            metrics_data['auc'] = hist_df['auc'].mean()
        if 'val_loss' in hist_df:
            metrics_data['val_loss'] = hist_df['val_loss'].mean()
        if 'val_binary_accuracy' in hist_df:
            metrics_data['val_binary_accuracy'] = hist_df['val_binary_accuracy'].mean()
        if 'val_auc' in hist_df:
            metrics_data['val_auc'] = hist_df['val_auc'].mean()
        if 'val_recall' in hist_df:
            metrics_data['val_recall'] = hist_df['val_recall'].mean()
        if 'lr' in hist_df:
            metrics_data['final_lr'] = hist_df['lr'].iloc[-1]
            metrics_data['mean_lr'] = hist_df['lr'].mean()
        if 'f1' in hist_df:
            metrics_data['final_f1'] = hist_df['f1'].iloc[-1]
            metrics_data['mean_f1'] = hist_df['f1'].mean()
        if 'best_f1_so_far' in hist_df:
            metrics_data['best_f1'] = hist_df['best_f1_so_far'].max()
        if 'best_threshold' in hist_df:
            metrics_data['best_threshold'] = hist_df['best_threshold'].iloc[-1]

        # Agregar métricas de precision y recall
        if 'precision' in hist_df:
            metrics_data['best_precision'] = hist_df['precision'].max()
            metrics_data['final_precision'] = hist_df['precision'].iloc[-1]
        if 'recall' in hist_df:
            metrics_data['best_recall'] = hist_df['recall'].max()
            metrics_data['final_recall'] = hist_df['recall'].iloc[-1]
        metrics_df = pd.DataFrame.from_records([metrics_data])
        with open(metrics_file_name, mode='w') as f:
            metrics_df.to_json(f)
        print(f"Per-epoch metrics saved to {plots_file_name} and aggregated metrics saved to {metrics_file_name}")

    print(f"TensorBoard logs written to: {log_dir}. Launch with: tensorboard --logdir {log_dir}")
