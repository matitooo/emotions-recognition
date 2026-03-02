import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm


def train_model(model, optimizer, train_loader, val_loader, criterion, n_epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)

    scaler = GradScaler(enabled=torch.cuda.is_available())

    
    torch.backends.cudnn.benchmark = True

    for epoch in tqdm(range(n_epochs)):

        
        model.train()
        train_loss = 0.0
        total_train = 0

        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            total_train += images.size(0)

        train_loss /= total_train

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f}")

        if epoch % 10 == 0:

            model.eval()
            val_loss = 0.0
            total_val = 0

            with torch.no_grad():
                for images, labels in val_loader:

                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    total_val += images.size(0)

            val_loss /= total_val

            print(f"Validation Loss: {val_loss:.4f}")

    return model