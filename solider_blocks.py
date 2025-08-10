import torch
import torch.nn as nn
import torch.nn.functional as F
from solider_config import SOLIDERConfig

class SOLIDERCNNBlock(nn.Module):
    """
    ResNet block with integrated semantic control.
    Supports Bottleneck (1x1,3x3,1x1) and BasicBlock (3x3,3x3).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
        config: SOLIDERConfig = None,
        bottleneck: bool = True,
        bottleneck_mid_channels: int = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = SOLIDERConfig()
        self.config = config
        embed_dim = config.semantic_embed_dim
        self.bottleneck = bottleneck

        self.relu = nn.ReLU(inplace=True)

        if self.bottleneck:
            assert bottleneck_mid_channels is not None, "bottleneck_mid_channels must be provided for bottleneck blocks"
            # 1x1 reduce
            self.conv1 = nn.Conv2d(in_channels, bottleneck_mid_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(bottleneck_mid_channels)
            # 3x3
            self.conv2 = nn.Conv2d(bottleneck_mid_channels, bottleneck_mid_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(bottleneck_mid_channels)
            # 1x1 expand
            self.conv3 = nn.Conv2d(bottleneck_mid_channels, out_channels, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            # Basic block
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample for identity path if needed
        if downsample is not None:
            self.downsample = downsample
        elif stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

        # Semantic control embeddings applied on output channels
        self.semantic_embed_w = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, out_channels),
            nn.Softplus(),
        )
        self.semantic_embed_b = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, out_channels),
        )

        if config.freeze_semantic_embeddings:
            for p in self.semantic_embed_w.parameters():
                p.requires_grad = False
            for p in self.semantic_embed_b.parameters():
                p.requires_grad = False

        # Optional dynamic alignment layer as a last-resort safeguard
        self._align_conv = None
        self._align_bn = None

    def forward(self, x: torch.Tensor, lambda_val: float = 0.5) -> torch.Tensor:
        identity = x

        if self.bottleneck:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if lambda_val > 0:
            lambda_tensor = torch.tensor([[float(lambda_val)]], device=x.device, dtype=torch.float32)
            w = self.semantic_embed_w(lambda_tensor).view(1, -1, 1, 1)
            b = self.semantic_embed_b(lambda_tensor).view(1, -1, 1, 1)
            out = out * w + b

        # Final safeguard: ensure channel alignment before residual add
        if out.shape[1] != identity.shape[1]:
            # Align out to identity's channels
            if self._align_conv is None or (
                self._align_conv.in_channels != out.shape[1] or
                self._align_conv.out_channels != identity.shape[1]
            ):
                self._align_conv = nn.Conv2d(out.shape[1], identity.shape[1], kernel_size=1, bias=False).to(out.device)
                self._align_bn = nn.BatchNorm2d(identity.shape[1]).to(out.device)
            out = self._align_bn(self._align_conv(out))

        out = out + identity
        out = self.relu(out)
        return out

class SOLIDERStage(nn.Module):
    """
    Wrapper for ResNet stages to propagate lambda_val through blocks.
    """
    def __init__(self, stage: nn.Module, config: SOLIDERConfig = None):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        # Convert each block in the stage to a SOLIDERCNNBlock, mirroring structure
        for block in stage:
            in_channels = block.conv1.in_channels
            if hasattr(block, 'conv3'):
                # Bottleneck
                bottleneck = True
                bottleneck_mid_channels = block.conv2.out_channels
                out_channels = block.conv3.out_channels
                stride = block.conv2.stride[0] if isinstance(block.conv2.stride, tuple) else block.conv2.stride
            else:
                # Basic
                bottleneck = False
                bottleneck_mid_channels = None
                out_channels = block.conv2.out_channels
                stride = block.conv1.stride[0] if isinstance(block.conv1.stride, tuple) else block.conv1.stride

            downsample = block.downsample if hasattr(block, 'downsample') else None

            solider_block = SOLIDERCNNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                config=config,
                bottleneck=bottleneck,
                bottleneck_mid_channels=bottleneck_mid_channels,
            )
            self.blocks.append(solider_block)
    
    def forward(self, x: torch.Tensor, lambda_val: float = 0.5) -> torch.Tensor:
        """Forward pass propagating lambda_val through all blocks."""
        for block in self.blocks:
            x = block(x, lambda_val)
        return x
