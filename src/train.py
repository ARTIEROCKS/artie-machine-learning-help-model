import pandas as pd
import numpy as np
import math
import sys
import yaml
np.random.seed(0)
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Masking, TimeDistributed
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
def load_time_series(df, max_time_steps, columns, mask_value, percentage_train):
    # Reading the data and separating the features from the target
    df_X = df.drop(axis=1, columns=["request_help"])
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
def generate_model(shape, data, mask_value, lstm_units, return_sequences=False, second_lstm_layer=False,
                   use_dropout=False, dropout_value=0.5):
    model = Sequential()
    model.add(Masking(mask_value=mask_value, input_shape=shape))
    model.add(LSTM(lstm_units, return_sequences=return_sequences))

    # If we decide to add a dropout layer
    if use_dropout:
        model.add(Dropout(dropout_value))

    # If we decide to add a second lstm layer
    if second_lstm_layer:
        model.add(LSTM(lstm_units, return_sequences=return_sequences))

    # If we decide to add a dropout layer
    if use_dropout:
        model.add(Dropout(dropout_value))

    model.add(TimeDistributed(Dense(1)))

    # Because we are in a binary problem, we use the binary cross entropy
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["binary_accuracy"])

    return model


# Loading the parameters
params_file = sys.argv[1]
input_csv_file = sys.argv[2]
output_model_file = sys.argv[3]
plots_file_name = sys.argv[4]

with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

mask_value = params['model']['mask_value']
percentage_train_size = params['model']['percentage_train_size']

lstm_units = params['model']['lstm_units']
return_sequences = params['model']['return_sequences']
second_lstm_layer = params['model']['second_lstm_layer']
use_dropouts = params['model']['use_dropouts']
dropout_value = params['model']['dropout_value']

training_epochs = params['model']['training_epochs']
training_batch_size = params['model']['training_batch_size']

# Loads the data file and gets the maximum time steps and the number of columns
df, max_time_steps, columns = load(input_csv_file)

# Loading the training and tests sets and fill the data with the mask value until the max time steps has been reached
df, train_x, train_y, test_x, test_y = load_time_series(df,max_time_steps, columns, mask_value, percentage_train_size)

# Executing the training
shape = (None, train_x.shape[2])
model = generate_model(shape, train_x, mask_value, lstm_units, return_sequences, second_lstm_layer, use_dropouts, dropout_value)
history = model.fit(train_x, train_y, epochs=training_epochs, batch_size=training_batch_size, validation_data=(test_x, test_y), verbose=2, shuffle=False)

# Saving the model
model.save(output_model_file)

# Saving the plots

# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history.history)
print(hist_df)
with open(plots_file_name, mode='w') as f:
    hist_df.to_csv(f, index_label='epoch')