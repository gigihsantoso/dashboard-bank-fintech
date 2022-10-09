import streamlit as st
import helper as help
import tensorflow as tf

numeric_data = help.train().describe()

dataset_train_fc_iqr = help.train().copy()
dataset_train_fc_iqr = help.iqr(dataset_train_fc_iqr, numeric_data)


dataset_train_model = dataset_train_fc_iqr.copy()
dataset_train_model['target'] = dataset_train_model.loc[:, 'y'].apply(lambda x : 0 if x == "no" else 1)
dataset_train_model.drop(columns=["duration", "day", "pdays", "previous", "month", "contact", "poutcome", "y"], inplace=True)

train, val = help.dataframe_split(dataset_train_model, ratio = 0.2, shuffle=1)

batch_size = 128
step = 256

train_ds = help.dataframe_imbalance_to_dataset(train, batch_size=batch_size, weights=[0.5, 0.5])
val_ds = help.dataframe_to_dataset(val, batch_size=batch_size)

bins= [10000,15000]
num_to_categorical_keys = ["campaign", "balance"]
num_keys = ["age"]
categorical_string_keys = ["job", "marital", "education"]
binary_string_keys = ["housing", "loan", "default"]
all_inputs = []
all_preprocessors = []

for categorical in binary_string_keys:
  input = tf.keras.Input(shape=(1,), name=categorical, dtype="string")
  cat_layer = help.categories_preprocessing(name=categorical, dataset=train_ds, dtype="string", step=step, output_mode="one_hot")
  cat_preprocessor = cat_layer(input)
  all_inputs.append(input)
  all_preprocessors.append(cat_preprocessor)

for categorical in categorical_string_keys:
  input = tf.keras.Input(shape=(1,), name=categorical, dtype="string")
  cat_layer = help.categories_preprocessing(name=categorical, dataset=train_ds, dtype="string", step=step, output_mode="one_hot")
  cat_preprocessor = cat_layer(input)
  all_inputs.append(input)
  all_preprocessors.append(cat_preprocessor)

for num_cat, bin in zip(num_to_categorical_keys, bins):
  input = tf.keras.Input(shape=(1,), name=num_cat, dtype="int64")
  num_cat_layer = help.numeric_to_categories_preprocessing(name=num_cat, dataset=train_ds, step=step, num_bins=bin, output_mode="one_hot")
  num_cat_preprocessor = num_cat_layer(input)
  all_inputs.append(input)
  all_preprocessors.append(num_cat_preprocessor)

for num_cat in num_keys:
  input = tf.keras.Input(shape=(1,), name=num_cat, dtype="int64")
  num_layer = help.numeric_preprocessing(name=num_cat, dataset=train_ds, step=step)
  num_preprocessor = num_layer(input)
  all_inputs.append(input)
  all_preprocessors.append(num_preprocessor)

all_features = tf.keras.layers.concatenate(all_preprocessors)
x = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.L2(0.0001))(all_features)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(x)
model = tf.keras.Model(all_inputs, output)  

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.SensitivityAtSpecificity(0.5, name='specificity')
]
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)

loss = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
val_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
history = model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=[loss, val_loss], steps_per_epoch=512)