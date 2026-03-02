import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEmotions(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNEmotions, self).__init__()

        # ----------- Convolutional Block 1 -----------
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2)
        self.bn1   = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop1 = nn.Dropout(0.25)

        # ----------- Convolutional Block 2 -----------
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop2 = nn.Dropout(0.25)

        # ----------- Convolutional Block 3 -----------
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(384)
        self.drop3 = nn.Dropout(0.25)

        # ----------- Convolutional Block 4 -----------
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(384)
        self.drop4 = nn.Dropout(0.25)

        # ----------- Convolutional Block 5 -----------
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop5 = nn.Dropout(0.25)

        # Adaptive pooling per input 48x48
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # ----------- Fully Connected -----------
        self.fc1 = nn.Linear(256 * 2 * 2, 4096)
        self.bn6 = nn.BatchNorm1d(4096)
        self.drop6 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        self.drop7 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(4096, num_classes)

        self._initialize_weights()


    def forward(self, x):

        # Block 1
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.drop3(x)

        # Block 4
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.drop4(x)

        # Block 5
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = self.pool5(x)
        x = self.drop5(x)

        # Adaptive Pool
        x = self.adaptive_pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully Connected
        x = F.relu(self.fc1(x))
        x = self.bn6(x)
        x = self.drop6(x)

        x = F.relu(self.fc2(x))
        x = self.bn7(x)
        x = self.drop7(x)

        x = self.fc3(x)

        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)