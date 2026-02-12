from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm

def train_model(model,optimizer,train_dataloader,val_dataloader,criterion,n_epochs):
    for epoch in tqdm(range(n_epochs)):
        model.train()
        optimizer.zero_grad()
        train_loss = 0 
        total_train = 0 
        for images, labels in train_dataloader:
            out = model(images)
            loss = criterion(out,labels)
            loss.backward()
                               
            train_loss += loss.item() * images.size(0)
            total_train += images.size(0)
            optimizer.step()
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss/total_train:.4f}")    
        if epoch%10 == 0:
            model.eval()
            val_loss = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_dataloader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    total += images.size(0)

            val_loss /= total
            print(f"Epoch {epoch+1}/{n_epochs} | Validation Loss: {val_loss:.4f}")

    return model