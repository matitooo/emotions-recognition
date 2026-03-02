import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEmotions(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNEmotions, self).__init__()

        # ----------- Block 1 -----------
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(2, 2)

        # ----------- Block 2 -----------
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(2, 2)

        # ----------- Block 3 -----------
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(256)

        self.pool3 = nn.MaxPool2d(2, 2)

        # After pooling:
        # 48 → 24 → 12 → 6
        # Final feature map: (B, 256, 6, 6)

        # ----------- Fully Connected -----------
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256 * 6 * 6, 128)
        self.bn7 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.0)  # rate=0 as in paper

        self.fc2 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def forward(self, x):

        # Block 1
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3
        x = F.elu(self.bn5(self.conv5(x)))
        x = F.elu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Fully Connected
        x = self.flatten(x)

        x = F.elu(self.bn7(self.fc1(x)))
        x = self.dropout(x)

        x = self.fc2(x)  # no softmax (use CrossEntropyLoss)

        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)