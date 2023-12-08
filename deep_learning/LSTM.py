import IPython
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import gc

# Function to extract training, validation, and test sets
def extract_train_val_test(dataframe, identifier_map, num_validation=2, num_test=2, warmup_period=warmup, df_id_map=df_id):
    dataframe = dataframe.copy() if 'state' not in dataframe.columns else dataframe[(dataframe['id_map'] == identifier_map) & (dataframe['state'] != -1)].copy()
    dataframe = augment_dataframe(dataframe, identifier_map)
    columns_to_use = ['id_map', 'step', 'time'] + EXP + (OBJ if 'state' in dataframe.columns else [])
    dataframe = dataframe[columns_to_use]
    IPython.display.clear_output()
    
    total_nights = df_id_map[df_id_map['id_map'] == identifier_map]['night'].iat[0]
    train_size = total_nights - num_test - num_validation
    event_steps = df_event[(df_event['id_map'] == identifier_map) & (df_event['step'] != -1)]['step']

    test_split_step = calculate_split_step(event_steps, train_size + num_validation, num_test)
    val_split_step = calculate_split_step(event_steps, train_size, num_validation, default_split=test_split_step)
    
    training_set = create_subset(dataframe, 0, val_split_step, warmup_period, identifier_map, columns_to_use)
    validation_set = create_subset(dataframe, val_split_step, test_split_step, warmup_period, identifier_map, columns_to_use) if num_validation else dataframe.iloc[:0]
    test_set = create_subset(dataframe, test_split_step, None, warmup_period, identifier_map, columns_to_use) if num_test else dataframe.iloc[:0]
    
    print(f'Total Nights: {total_nights}, Training Size: {train_size}, Validation Size: {num_validation}, Test Size: {num_test}')
    
    return training_set, validation_set, test_set

def calculate_split_step(steps, position, is_active, default_split=None):
    if is_active:
        end = steps.iat[position * 2]
        start = steps.iat[position * 2 + 1]
        return (end + start) // 2
    return default_split

def create_subset(df, start_step, end_step, warmup, id_map, columns):
    subset = df if end_step is None else df[(df['step'] >= start_step) & (df['step'] < end_step)].reset_index(drop=True)
    return augment_dataframe(subset, id_map, warmup, columns)

# Example usage
id_map = TRAIN_ID
train, val, test = extract_train_val_test(df, id_map, 5, 5)

# Preprocessing datasets
ds_train = make_dataset(std_df(train))
ds_val = make_dataset(std_df(val))

# Model Definition
model = Sequential([
    LSTM(64, return_sequences=False),
    Dropout(0.7),
    Dense(units=1, activation='sigmoid')
])

# Callbacks
checkpoint_callback = ModelCheckpoint(
    filepath=best_model_path,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min')

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss=BinaryCrossentropy(from_logits=False),
              metrics=["accuracy"])

# Training the model
# %%time
history = model.fit(ds_train, epochs=20,
                    validation_data=ds_val,
                    callbacks=[early_stopping, checkpoint_callback],
                    class_weight=class_weight(df[df['id_map']==id_map]))

gc.collect()

# Loading best model and making predictions
best_model = tf.keras.models.load_model(best_model_path)
test, _, _ = extract_train_val_test(df, TEST_ID, 0, 0, warmup)
predictions = prediction(best_model, test)


def augment_dataframe(df, id_map, warmup=5, columns=None):
    augmented_df = df.copy()
    if columns:
        augmented_df = augmented_df[columns]
    return augmented_df

def make_dataset(df):
    dataset = df.copy()
    if 'sleep_stage' in df.columns:
        y = to_categorical(df['sleep_stage'])
        X = df.drop('sleep_stage', axis=1)
        return X, y
    return df

def std_df(df):
    scaler = StandardScaler()
    standardized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return standardized_df

def class_weight(df):
    class_weights = {
        0: 1.0,  # Weight for class 0
        1: 1.0   # Weight for class 1
        # Add more weights for each class if necessary
    }
    return class_weights

def prediction(model, test_data):
    predictions = model.predict(test_data)
    return predictions
