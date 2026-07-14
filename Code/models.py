import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut (跳层连接)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Encoder(nn.Module):

    def __init__(self, num_tags, feature_dim=128):
        super(Encoder, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(num_tags, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = ResBlock(32, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 128, stride=1)  # 保持维度，增加深度


        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.adaptive_pool(x)
        x = x.squeeze(-1)

        features = self.fc(x)
        return features


class ProjectionHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TagCLR(nn.Module):

    def __init__(self, num_tags, modalities, feature_dim=128, projection_dim=128):
        super(TagCLR, self).__init__()
        self.modalities = modalities
        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        for m in modalities:
            self.encoders[m] = Encoder(num_tags, feature_dim)
            self.projectors[m] = ProjectionHead(feature_dim, feature_dim, projection_dim)

    def forward(self, view1, view2):
        mod1 = self.modalities[0]
        mod2 = self.modalities[-1]

        h1 = self.encoders[mod1](view1)
        h2 = self.encoders[mod2](view2)

        z1 = self.projectors[mod1](h1)
        z2 = self.projectors[mod2](h2)
        return z1, z2


class ClassificationModel(nn.Module):

    def __init__(self, num_tags, num_classes, modalities, feature_dim=128):
        super(ClassificationModel, self).__init__()
        self.modalities = modalities

        # 同样使用Encoder
        self.encoders = nn.ModuleDict()
        for m in modalities:
            self.encoders[m] = Encoder(num_tags, feature_dim)

        input_dim = feature_dim * len(modalities)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, inputs_dict):
        features = []
        for m in self.modalities:
            features.append(self.encoders[m](inputs_dict[m]))

        if len(features) > 1:
            combined = torch.cat(features, dim=1)
        else:
            combined = features[0]
        return self.classifier(combined)