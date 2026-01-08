import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """
    Basic residual block for 3D volumes.
    """
    def __init__(self, c_in, c_out, stride=1, norm="gn"):
        super().__init__()

        Norm = nn.BatchNorm3d if norm == "bn" else nn.GroupNorm

        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = Norm(c_out) if norm == "bn" else Norm(num_groups=16, num_channels=c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = Norm(c_out) if norm == "bn" else Norm(num_groups=16, num_channels=c_out)

        self.proj = None

        if stride != 1 or c_in != c_out:
            self.proj = nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                (Norm(c_out) if norm == "bn" else Norm(num_groups=16, num_channels=c_out)),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class FeatureExtractorFE(nn.Module):
    """
    Input:  x [B, 1, D, H, W]
    Output: y [B, 512, D//16]

    Downsamples by /16 along D using 4 stride-2 operations:
      stem conv stride2  -> /2
      maxpool stride2    -> /4
      layer2 stride2     -> /8
      layer3 stride2     -> /16
    Then global-average-pool over H,W (cross-section), keep depth tokens.
    """
    def __init__(self, in_ch=1, norm="gn"):
        super().__init__()
        # Stem: /4
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),  # /2
            nn.GroupNorm(16, 64) if norm == "gn" else nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),                      # /4
        )

        # Residual stages (ResNet-like)
        self.layer1 = ResBlock3D(64,  64,  stride=1, norm=norm)   # keep
        self.layer2 = ResBlock3D(64,  128, stride=2, norm=norm)   # /8
        self.layer3 = ResBlock3D(128, 256, stride=2, norm=norm)   # /16
        self.layer4 = ResBlock3D(256, 512, stride=1, norm=norm)   # keep /16

    def forward(self, x):
        """
        x: [B,1,D,H,W]
        returns: [B,512,D//16]
        """
        x = self.stem(x)    # [B,64, D/4,  H/4,  W/4]
        x = self.layer1(x)  # [B,64, ...]
        x = self.layer2(x)  # [B,128, D/8,  H/8,  W/8]
        x = self.layer3(x)  # [B,256, D/16, H/16, W/16]
        x = self.layer4(x)  # [B,512, D/16, H/16, W/16]

        # Global average pool over cross-section (H,W), keep depth
        x = x.mean(dim=(-1, -2))   # [B,512,D/16]
        return x