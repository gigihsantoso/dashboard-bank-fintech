import zipfile
import pandas as pd
import tensorflow as tf
import matplotlib as plt

def train():
    data_url = "C:/Users/mrohman\Downloads/fintech_banking_dataset.zip"
    data_dir = zipfile.ZipFile(data_url, 'r')
    data_dir.extractall("MyFile")
    data_dir.close()

    return pd.read_csv('MyFile/train.csv', delimiter=";")
    
def test():
    data_url = "C:/Users/mrohman\Downloads/fintech_banking_dataset.zip"
    data_dir = zipfile.ZipFile(data_url, 'r')
    data_dir.extractall("MyFile")
    data_dir.close()

    return pd.read_csv('MyFile/test.csv', delimiter=";")
    
def dataframe_split(data, ratio= 0.2, shuffle= 0):
  len_ = int(len(data) - (ratio*len(data)))
  if shuffle == 0 :
    part_  = data.iloc[:int(1 + len_)]
    rest_part_ = data.drop(part_.index)
  elif shuffle == 1:
    part_ = data.sample(frac = 1.0 - ratio)
    rest_part_ = data.drop(part_.index)
  return (part_, rest_part_)

def dataframe_to_dataset(data, batch_size=32):
  df = data.copy()
  labels = df.pop('target')
  df = {key:  tf.expand_dims(value,axis=1) for key, value in df.items()}
  ds = tf.data.Dataset.from_tensor_slices((df, labels))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds

def dataframe_imbalance_to_dataset(data, 
                                   batch_size=32, 
                                   shuffle=32, 
                                   weights=[0.5, 0.5]):
  df = data.copy()
  labels = df.pop('target')
  df = {key:  tf.expand_dims(value,axis=1) for key, value in df.items()}
  data = tf.data.Dataset.from_tensor_slices((dict(df), labels)).batch(batch_size)

  no_ds = (data
          .unbatch()
          .filter(lambda features, label: label==0)
          .shuffle(shuffle)
          .repeat())
  yes_ds = (data
            .unbatch()
            .filter(lambda features, label: label==1)
            .shuffle(shuffle)
            .repeat())
  
  ds = tf.data.Dataset.sample_from_datasets([no_ds, yes_ds], weights=weights)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds

def numeric_preprocessing(name, dataset, step):
  feature_dataset = dataset.map(lambda x,y: x[name])
  layer = tf.keras.layers.Normalization()
  layer.adapt(feature_dataset, steps=step)

  return layer

def numeric_to_categories_preprocessing(name, dataset, step, num_bins, output_mode="multi_hot"):
  feature_dataset = dataset.map(lambda x,y: x[name])
  layer = tf.keras.layers.Discretization(num_bins=num_bins, epsilon=0.00001)
  layer.adapt(feature_dataset, steps=step)
  
  category_encoding = tf.keras.layers.CategoryEncoding(num_tokens=num_bins, output_mode=output_mode)
  return lambda feature:  category_encoding(layer(feature))

def categories_preprocessing(name, dataset, dtype, step=1, output_mode="multi_hot"):
  if(dtype == "int64"):
    lookup = tf.keras.layers.IntegerLookup()
  else:
    lookup = tf.keras.layers.StringLookup()
  feature_dataset = dataset.map(lambda x,y: x[name])
  lookup.adapt(feature_dataset, steps=step)

  category_encoding = tf.keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size(), output_mode=output_mode)
  return lambda feature:  category_encoding(lookup(feature))

def plot_metric(x, y, history, metric):
  fig, ax = plt.subplots(x, y, figsize=(15,10))
  epochs = range(len(history.history[metric[0]]))
  for metric, ax in zip(metric, ax.flatten()):
    ax.plot(epochs, history.history[metric], 'r', label='train')
    ax.plot(epochs, history.history['val_'+metric], 'b', ls="--", label='val')
    ax.set_ylabel(metric)
    ax.set_xlabel('epoch')
    ax.legend()  
  plt.show()

def iqr(dataset_train_fc_iqr, numeric_data):
  Q1_balance = numeric_data.loc['25%', 'balance']
  Q3_balance = numeric_data.loc['75%', 'balance']
  IQR_balance = Q3_balance - Q1_balance
  outlier_upper_balance = Q3_balance +  (1.5 * IQR_balance)
  outlier_lower_balance = Q1_balance -  (1.5 * IQR_balance)

  Q1_campaign = numeric_data.loc['25%', 'campaign']
  Q3_campaign = numeric_data.loc['75%', 'campaign']
  IQR_campaign = Q3_campaign - Q1_campaign
  outlier_upper_campaign = Q3_campaign +  (1.5 * IQR_campaign)
  outlier_lower_campaign = Q1_campaign -  (1.5 * IQR_campaign)

  Q1_pdays = numeric_data.loc['25%', 'pdays']
  Q3_pdays = numeric_data.loc['75%', 'pdays']
  IQR_pdays = Q3_pdays - Q1_pdays
  outlier_upper_pdays = Q3_pdays +  (1.5 * IQR_pdays)
  outlier_lower_pdays = Q1_pdays -  (1.5 * IQR_pdays)

  return dataset_train_fc_iqr.loc[
                                              dataset_train_fc_iqr['balance'].le(outlier_upper_balance)&
                                              dataset_train_fc_iqr['balance'].ge(outlier_lower_balance)&
                                              dataset_train_fc_iqr['campaign'].le(outlier_upper_campaign)&
                                              dataset_train_fc_iqr['campaign'].ge(outlier_lower_campaign)&
                                              dataset_train_fc_iqr['pdays'].le(outlier_upper_pdays)&
                                              dataset_train_fc_iqr['pdays'].ge(outlier_lower_pdays)
  ]