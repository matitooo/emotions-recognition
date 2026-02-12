import cv2
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
def data_preprocessing(dataset):
    X = []
    y = []
    for i in range(dataset.shape[0]):
        emotion = dataset.iloc[i]['label']
        path = dataset.iloc[i]['path']
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img_resized=cv2.resize(img,(48,48))
        X.append(img_resized)
        y.append(emotion)
    X = np.array(X).astype('float32')
    X = X/255.0
    y = np.array(y)
    return X,y
    
def tensor_batch_processing(X,y,batch_size):
    X_small,X_res,y_small,y_res = train_test_split(X,y,test_size=0.30,shuffle=True,random_state=42)
    X_train,X_val,y_train,y_val = train_test_split(X_small,y_small,test_size=0.2)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val   = torch.tensor(y_val, dtype=torch.long)
    
    X_train = X_train.unsqueeze(1)
    X_val   = X_val.unsqueeze(1)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

    val_loader = DataLoader(val_dataset,
                        shuffle=False)
    
    return train_loader,val_loader