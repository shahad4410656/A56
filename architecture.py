import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 384, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((3, 3))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),

            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 200)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
