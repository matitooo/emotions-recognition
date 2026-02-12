import pandas as pd
from utils import data_preprocessing,tensor_batch_processing
from sklearn.model_selection import train_test_split
from model import CNNEmotions
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model_utils import train_model
train_path = 'data/train/train.csv'
df_train = pd.read_csv(train_path)
X,y = data_preprocessing(df_train)
train_loader, val_loader = tensor_batch_processing(X,y,batch_size=64)
model = CNNEmotions()
optim = optimizer = Adam(model.parameters(), lr=0.001)
n_epochs = 100
criterion = CrossEntropyLoss()
train_model(model,optim,train_loader,val_loader,criterion,n_epochs)

