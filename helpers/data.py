import pandas as pd
import numpy as np
import torch

class Data:
  def read_data(file):
      df = pd.read_csv(file).sort_values(by=['Date'], ascending=True)
      df['pct_change'] = df['Adj Close'].pct_change(1)
      df['change'] = df['Adj Close'].diff()
      return df

  def split_data(df, training_size = 0.7):
      size = df.shape[0]
      train = int(training_size * size)
      val = size - train
      Xtrain = df.iloc[:train, :]
      Xval = df.iloc[train:, :]

      return Xtrain, Xval

  def get_state(data, start, days_back):
      end = start - days_back
      return torch.tensor(data[end:start]['Close'].values, dtype=torch.float), torch.sigmoid(torch.tensor(data[end:start]['change'].values, dtype=torch.float))

