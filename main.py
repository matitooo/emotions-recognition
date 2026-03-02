import os
import random
import numpy as np
import pandas as pd
import torch

from utils import data_preprocessing, tensor_batch_processing
from model import CNNEmotions
from model_utils import train_model

from torch.optim import Adam
from torch.nn import CrossEntropyLoss


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    train_path = "data/train/train.csv"
    df_train = pd.read_csv(train_path)

    X, y = data_preprocessing(df_train)

    train_loader, val_loader = tensor_batch_processing(
        X,
        y,
        batch_size=64,
        num_workers=2
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")


    model = CNNEmotions().to(device)

    # PyTorch 2.x acceleration
    if torch.__version__ >= "2.0":
        model = torch.compile(model)

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()

    n_epochs = 100


    model = train_model(
        model,
        optimizer,
        train_loader,
        val_loader,
        criterion,
        n_epochs
    )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/cnn_emotions.pt")

    print("Training completed and model saved.")



if __name__ == "__main__":
    main()