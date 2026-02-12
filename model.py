import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEmotions(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNEmotions, self).__init__()

        # ----------- Convolutional Block 1 -----------
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        # ----------- Convolutional Block 2 -----------
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        # ----------- Convolutional Block 3 -----------
        self.conv4 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)

        # ----------- Convolutional Block 4 -----------
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(0.25)

        # ----------- Convolutional Block 5 -----------
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.drop5 = nn.Dropout(0.25)

        # After last pooling:
        # Input: 48x48
        # → 24 → 12 → 6 → 3 → 1
        # Final feature map: (B, 512, 1, 1)

        # ----------- Fully Connected -----------
        self.fc1 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.drop6 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.drop7 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):

        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.conv4(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4
        x = F.relu(self.conv5(x))
        x = self.bn4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        # Block 5
        x = F.relu(self.conv6(x))
        x = self.bn5(x)
        x = self.pool5(x)
        x = self.drop5(x)

        # Flatten
        x = torch.flatten(x, 1)  # (B, 512)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.bn6(x)
        x = self.drop6(x)

        x = F.relu(self.fc2(x))
        x = self.bn7(x)
        x = self.drop7(x)

        x = self.fc3(x)  # no softmax (use CrossEntropyLoss)

        return x
