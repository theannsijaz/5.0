import torch
import torch.nn as nn
import torch.nn.functional as F
from solider_config import SOLIDERConfig

class SOLIDERCNNBlock(nn.Module):
    """
    Enhanced ResNet block with integrated semantic control.
    Applies semantic modulation through learned embeddings directly in the block.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: nn.Module = None, config: SOLIDERConfig = None):
        super().__init__()
        
        # Get config or use defaults
        if config is None:
            config = SOLIDERConfig()
        
        self.config = config
        embed_dim = config.semantic_embed_dim
        
        # Standard ResNet block components
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        # Semantic control embeddings
        self.semantic_embed_w = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, out_channels),
            nn.Softplus()  # Ensures positive scaling
        )
        
        self.semantic_embed_b = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, out_channels)
        )
        
        # Freeze semantic embeddings if configured
        if config.freeze_semantic_embeddings:
            for param in self.semantic_embed_w.parameters():
                param.requires_grad = False
            for param in self.semantic_embed_b.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor, lambda_val: float = 0.5) -> torch.Tensor:
        identity = x
        
        # Standard ResNet forward pass
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsampling if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Apply semantic modulation
        if lambda_val > 0:  # Only apply if some semantic control is desired
            # Convert lambda to tensor
            lambda_tensor = torch.tensor([[float(lambda_val)]], 
                                       device=x.device, 
                                       dtype=torch.float32)
            
            # Get semantic modulation parameters
            w = self.semantic_embed_w(lambda_tensor)  # [1, C]
            b = self.semantic_embed_b(lambda_tensor)  # [1, C]
            
            # Reshape for broadcasting
            w = w.view(1, -1, 1, 1)
            b = b.view(1, -1, 1, 1)
            
            # Apply modulation: x * w + b
            out = out * w + b
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out

class SOLIDERStage(nn.Module):
    """
    Wrapper for ResNet stages to propagate lambda_val through blocks.
    """
    def __init__(self, stage: nn.Module, config: SOLIDERConfig = None):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        # Convert each block in the stage to a SOLIDERCNNBlock
        for block in stage:
            # Get block parameters
            in_channels = block.conv1.in_channels
            out_channels = block.conv2.out_channels
            stride = block.conv1.stride[0]
            downsample = block.downsample
            
            # Create SOLIDER block
            solider_block = SOLIDERCNNBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                config=config
            )
            self.blocks.append(solider_block)
    
    def forward(self, x: torch.Tensor, lambda_val: float = 0.5) -> torch.Tensor:
        """Forward pass propagating lambda_val through all blocks."""
        for block in self.blocks:
            x = block(x, lambda_val)
        return x