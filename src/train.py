import pandas as pd
import numpy as np
import math
import sys
import os
import yaml
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional

# Configurar semillas para reproducibilidad
np.random.seed(0)
tf.random.set_seed(0)
np.random.seed(1)


# Function to load the csv file
def load(file):

  # Reads the csv file
  df = pd.read_csv(file, delimiter=',')
  columns = len(df.columns) - 1
  max_time_steps = 1
  last_step_seconds = -1
  current_time_steps = 0

  # Reading all the rows
  for index, row in df.iterrows():

    # Get the "time" of the current row
    current_seconds = df["total_seconds"][index]

    # If the last step seconds is greater than the current one, we
    if last_step_seconds > current_seconds:
      current_time_steps = 0

    current_time_steps += 1

    # If the current time steps are greater than the max, we set the max time steps
    if current_time_steps > max_time_steps:
      max_time_steps = current_time_steps

    last_step_seconds = current_seconds

  return df,max_time_steps,columns


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

    # Determines the columns to drop
    apted_columns = ["request_help", "apted_distance", "tree_grade"]
    artie_columns = ["request_help", "solution_distance_family_distance", "solution_distance_element_distance",
                     "solution_distance_position_distance", "solution_distance_input_distance",
                     "solution_distance_total_distance", "grade"]

    if distance_calculation_type.lower() == 'artie':
        cols_to_drop = [col for col in apted_columns if col in df.columns]
    else:
        cols_to_drop = [col for col in artie_columns if col in df.columns]

    # Removes the columns that we do not need
    df_X = df.drop(axis=1, columns=cols_to_drop)

    # Updates the columns
    columns = df_X.shape[1]

    df_y = df["request_help"]

    last_step_seconds = -1

    time_steps_x = None
    time_steps_y = None

    sample_x = None
    sample_y = None

    for index, row in df_X.iterrows():

        current_seconds = df_X["total_seconds"][index]

        # If the time steps of x is none, we create a new np array
        if time_steps_x is None:
            time_steps_x = np.array([row])
            time_steps_y = np.array([df_y.iloc[[index]]])
        else:
            time_steps_x = np.vstack([time_steps_x, row])
            time_steps_y = np.vstack([time_steps_y, df_y.iloc[[index]]])

        # If the last step seconds are greater than the current seconds
        # we add a new sample block
        if last_step_seconds > current_seconds:

            # We complete the time series with the maximum and the number of columns
            time_steps_x, time_steps_y = padding_masking(time_steps_x, time_steps_y, max_time_steps, columns,
                                                         mask_value)

            if sample_x is None:
                sample_x = np.array([time_steps_x])
                sample_y = np.array([time_steps_y])
            else:
                sample_x = np.vstack([sample_x, [time_steps_x]])
                sample_y = np.vstack([sample_y, [time_steps_y]])

            time_steps_x = None
            time_steps_y = None

        # We get the current seconds as the new last step seconds
        last_step_seconds = current_seconds

    # We calculate the train size in base of the percentage
    train_size = math.floor(sample_x.shape[0] * percentage_train / 100)

    train_x = sample_x[:train_size]
    train_y = sample_y[:train_size]

    test_x = sample_x[train_size + 1:]
    test_y = sample_y[train_size + 1:]

    return df, train_x, train_y, test_x, test_y


# Function to generate the model
def generate_model(shape, mask_value, lstm_units, return_sequences=False, second_lstm_layer=False, use_dropout=False,
                   dropout_value=0.5, use_bidirectional=True):

    # Crear el modelo usando la API funcional para manejar mejor las máscaras
    inputs = tf.keras.Input(shape=shape)
    # Crear capa de máscara
    masked = layers.Masking(mask_value=mask_value)(inputs)

    # Si queremos agregar una segunda capa LSTM
    if second_lstm_layer:
        if use_bidirectional:
            x = Bidirectional(layers.LSTM(lstm_units, activation='sigmoid', return_sequences=True))(masked)
        else:
            x = layers.LSTM(lstm_units, activation='sigmoid', return_sequences=True)(masked)
        # Si queremos agregar dropout después de la primera capa LSTM
        if use_dropout:
            x = layers.Dropout(dropout_value)(x)
    else:
        x = masked

    # Agregar la capa LSTM principal (siempre presente)
    if use_bidirectional:
        x = Bidirectional(layers.LSTM(lstm_units, activation='sigmoid', return_sequences=return_sequences))(x)
    else:
        x = layers.LSTM(lstm_units, activation='sigmoid', return_sequences=return_sequences)(x)

    # Si queremos agregar dropout después de la capa LSTM principal
    if use_dropout:
        x = layers.Dropout(dropout_value)(x)

    # Capa de salida - en lugar de TimeDistributed, usamos una capa Dense con la forma adecuada
    if return_sequences:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(1, activation='sigmoid')(x)

    # Crear el modelo
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),  # Valor predeterminado, se sobrescribirá después
        metrics=['binary_accuracy', 
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc', curve='PR')]
    )

    return model

# Function to compute sample weights for time series data
def compute_sample_weights_for_time_series(train_y, mask_value, class_weights):
    """
    Compute sample weights for each time step in the time series data
    Versión optimizada usando operaciones vectorizadas
    """
    # Inicializar pesos con unos
    sample_weights = np.ones_like(train_y, dtype=float)

    # Convertir train_y a un array numpy si no lo es ya
    if not isinstance(train_y, np.ndarray):
        train_y = np.array(train_y)

    # Crear máscara para valores enmascarados
    mask = (train_y == mask_value)

    # Asignar peso 0 a valores enmascarados
    sample_weights[mask] = 0.0

    # Para cada clase, asignar el peso correspondiente
    for class_label, weight in class_weights.items():
        # Asegurarse de que class_label es un valor numérico
        class_label_value = float(class_label)
        # Crear máscara para esta clase (los valores que son igual a class_label y no son máscaras)
        class_mask = (train_y == class_label_value) & ~mask
        # Asignar peso
        sample_weights[class_mask] = weight

    return sample_weights



print(tf.__version__)

# Loading the parameters
params_file = sys.argv[1]
input_csv_file = sys.argv[2]
output_model_file = sys.argv[3]
plots_file_name = sys.argv[4]
metrics_file_name = sys.argv[5]

# Check if GPU should be used based on command line argument
use_gpu = False
if len(sys.argv) > 6:
    use_gpu = sys.argv[6].lower() == 'true'

with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

distance_calculation_type = params['model'].get('distance_calculation_type','artie') # ARTIE or APTED

mask_value = params['model'].get('mask_value', -1)  # Por defecto -1 para enmascarar valores
percentage_train_size = params['model'].get('percentage_train_size', 70)  # Por defecto 80% para entrenamiento
initial_learning_rate = params['model'].get('initial_learning_rate', 0.1)  # Por defecto 0.1

lstm_units = params['model'].get('lstm_units',256)  # Por defecto 256 unidades LSTM
return_sequences = params['model'].get('return_sequences',True)  # Por defecto activado
second_lstm_layer = params['model'].get('second_lstm_layer', True)  # Por defecto activado
use_dropouts = params['model'].get('use_dropouts', True)  # Por defecto activado
dropout_value = params['model'].get('dropout_value', 0.5)
use_bidirectional = params['model'].get('use_bidirectional', True)  # Por defecto activado

training_epochs = params['model'].get('training_epochs', 100)  # Por defecto 100 épocas
training_batch_size = params['model'].get('training_batch_size', 32)  # Por defecto 32 batch size
training_class_weights = params['model'].get('training_class_weights', True)  # Por defecto True, usar pesos de clase
training_early_stopping_patience = params['model'].get('training_early_stopping_patience', 10)  # Por defecto 10 épocas de paciencia
training_reduce_lr_patience = params['model'].get('training_reduce_lr_patience', 5)  # Por defecto 5 épocas de paciencia para reducir LR
training_reduce_lr_factor = params['model'].get('training_reduce_lr_factor', 0.1)  # Por defecto reducir LR por un factor de 0.1

show_summary = params['model'].get('show_summary', True)  # Por defecto mostrar el resumen del modelo

# list of all physical devices
print(tf.config.list_physical_devices())

# Configure TensorFlow to use Metal GPU if requested and available
if use_gpu:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Available GPUs: {gpus}")
        try:
            # Configuración específica para Metal GPU en Mac
            # No usamos memory growth para Metal ya que puede causar problemas
            # Enable all available GPUs
            tf.config.set_visible_devices(gpus, 'GPU')

            # Configuración óptima para Metal
            # Limitar el uso de memoria para evitar OOM
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )

            # Configuración para mejor rendimiento con Metal (si están disponibles)
            try:
                tf.config.optimizer.set_jit(False)  # Desactivar XLA que puede causar problemas con Metal
            except:
                pass

            print("Metal GPU enabled for training with safe settings")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
            use_gpu = False
    else:
        print("No GPUs found. Using CPU for training.")
        use_gpu = False
else:
    # Disable GPU usage
    tf.config.set_visible_devices([], 'GPU')
    print("GPU disabled for training. Using CPU.")

# Loads the data file and gets the maximum time steps and the number of columns
df, max_time_steps, columns = load(input_csv_file)

# Loading the training and tests sets and fill the data with the mask value until the max time steps has been reached
df, train_x, train_y, test_x, test_y = load_time_series(df, max_time_steps, columns, mask_value, percentage_train_size,
                                                        distance_calculation_type)

# Executing the training
shape = (None, train_x.shape[2])
model = generate_model(shape, mask_value, lstm_units, return_sequences, second_lstm_layer, use_dropouts, dropout_value, use_bidirectional)

# Log device information before training
print("Device configuration for training:")
print("- Visible devices:", tf.config.get_visible_devices())
print("- Using GPU:", use_gpu)

# Callback personalizado para imprimir el learning rate en cada época
class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Acceder al learning rate de manera compatible con versiones actuales de TF
        try:
            # Método moderno: usar get_config()
            lr = self.model.optimizer.get_config()['learning_rate']
            if hasattr(lr, 'numpy'):
                lr = lr.numpy()
        except (AttributeError, KeyError):
            # Método alternativo: intentar con _decayed_lr
            try:
                lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
            except (AttributeError, ValueError):
                # Último recurso: usar un valor fijo
                lr = "No disponible"
        print(f"\nLearning rate en época {epoch+1}: {lr}")

callbacks = []
if training_early_stopping_patience > 0:
    # Configurer callbacks for better performance and monitoring
    callbacks = [
        LearningRateLogger(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             factor=training_reduce_lr_factor,
                                             patience=training_reduce_lr_patience,
                                             min_lr=0.001,
                                             verbose=1),  # Verbose para mostrar los cambios en LR
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=training_early_stopping_patience,
                                         restore_best_weights=True)
    ]

# Use run_eagerly=True to avoid graph errors with symbolic tensors
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
    metrics=['binary_accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc', curve='PR')
             ],
    run_eagerly=True  # Esto soluciona muchos problemas de grafos
)

if training_class_weights:
    # Flatten the 3D array (samples, time_steps, 1) to 1D for class weight computation
    train_y_flat = train_y.reshape(-1)
    train_y_flat = train_y_flat[train_y_flat != mask_value]

    # Compute class weights to handle class imbalance
    classes = np.unique(train_y_flat)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_y_flat)

    # Create dictionary with class weights
    class_weights = dict(zip(classes, weights))
    print("Applied class weights:", class_weights)

    # Compute sample weights for time series data
    sample_weights = compute_sample_weights_for_time_series(train_y, mask_value, class_weights)

    # Ajust batch_size for better performance on GPU
    optimal_batch_size = 64 if use_gpu else training_batch_size

    print(f"Starting training with batch size: {optimal_batch_size}")
    print("Configured metrics:", [m.name if hasattr(m, 'name') else m for m in model.metrics])
    history = model.fit(train_x,
                        train_y,
                        epochs=training_epochs,
                        batch_size=optimal_batch_size,
                        validation_data=(test_x, test_y),
                        verbose=1,  # 1 = barra de progreso para cada época
                        shuffle=False,
                        callbacks=callbacks,
                        sample_weight=sample_weights)

else:

    # Adjustar batch_size para mejor rendimiento en GPU
    optimal_batch_size = 64 if use_gpu else training_batch_size

    print(f"Starting training with batch size: {optimal_batch_size}")
    print("Configured metrics:", [m.name if hasattr(m, 'name') else m for m in model.metrics])
    history = model.fit(train_x,
                        train_y,
                        epochs=training_epochs,
                        batch_size=optimal_batch_size,
                        validation_data=(test_x, test_y),
                        verbose=1,  # 1 = barra de progreso para cada época
                        shuffle=False,
                        callbacks=callbacks)

# Saving the model
print("\nTraining completed successfully. Saving model...")
try:
    # Intentar guardar en formato moderno .keras
    model.save(output_model_file, save_format='keras')
    print(f"Model saved to {output_model_file} in modern format")
except Exception as e:
    print(f"Error saving in modern format: {e}")
    # Fallback a formato HDF5
    model.save(output_model_file)
    print(f"Model saved to {output_model_file} in legacy HDF5 format")

# Check if the history is empty
if not history.history:
    print("WARNING: The training history is empty. It's possible the model has not been correctly trained.")
else:
    print(f"Modelo entrenado durante {len(history.history['loss'])} épocas.")
    # Check if precision and recall metrics are available
    if 'precision' in history.history and 'recall' in history.history:
        print(f"Final metrics: Precision: {history.history['precision'][-1]:.4f}, Recall: {history.history['recall'][-1]:.4f}")
    else:
        print("Available metrics:", list(history.history.keys()))
        print(f"Final metrics: binary_accuracy: {history.history['binary_accuracy'][-1]:.4f}")

#If we want to show the summary
if show_summary:
    print("\nModel summary:")
    model.summary()
    try:
        # Verificar si existe el directorio de imágenes
        if not os.path.exists('images'):
            os.makedirs('images')
        tf.keras.utils.plot_model(model, to_file='images/model.png', dpi=200)
        print("Model diagram saved to images/model.png")
    except Exception as e:
        print(f"Error generating model diagram: {e}")
        print("This is not critical for model training.")

# Saving the plots and metrics
# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history.history)

# Create a DataFrame for metrics
metrics_data = {'loss': hist_df['loss'].mean(), 'binary_accuracy': hist_df['binary_accuracy'].mean()}

# Add precision, recall, AUC and validation metrics if they exist
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

metrics_df = pd.DataFrame.from_records([metrics_data])

with open(plots_file_name, mode='w') as f:
    hist_df.to_csv(f, index_label='epoch')

with open(metrics_file_name, mode='w') as f:
    metrics_df.to_json(f)