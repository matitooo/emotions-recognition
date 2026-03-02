import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def data_preprocessing(dataset):
    X = []
    y = []

    for _, row in dataset.iterrows():
        emotion = row["label"]
        path = row["path"]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (48, 48))
        X.append(img)
        y.append(emotion)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int64)

    return X, y


def tensor_batch_processing(X, y, batch_size, num_workers=4):

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True
    )

    X_train = torch.tensor(X_train).unsqueeze(1)
    X_val   = torch.tensor(X_val).unsqueeze(1)

    y_train = torch.tensor(y_train)
    y_val   = torch.tensor(y_val)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader