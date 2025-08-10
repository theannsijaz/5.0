from dataclasses import dataclass
from typing import Literal, Tuple

@dataclass
class SOLIDERConfig:
    """Configuration for SOLIDER model and training."""
    
    # Model Architecture
    feature_dim: int = 2048
    semantic_embed_dim: int = 128  # Dimension for semantic embeddings
    num_semantic_parts: int = 3
    
    # Semantic Control
    semantic_weight: float = 0.5  # Weight for semantic loss
    lambda_distribution: Literal['binomial', 'beta', 'uniform'] = 'binomial'
    freeze_semantic_embeddings: bool = True
    
    # Semantic Parts (y-coordinate ranges)
    upper_body_range: Tuple[float, float] = (0.0, 0.40)  # Head, chest, arms
    lower_body_range: Tuple[float, float] = (0.40, 0.75)  # Waist, thighs
    shoes_range: Tuple[float, float] = (0.75, 1.0)  # Calves, feet
    
    # Training Stages
    stage1_epochs: int = 50  # FIDI-only stage
    stage2_epochs: int = 50  # SOLIDER stage
    stage2_freeze_epochs: int = 5  # Freeze backbone epochs in stage 2
    
    # Optimization
    learning_rate: float = 3e-4
    stage2_learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    
    # Loss Weights
    fidi_weight_start: float = 0.6
    fidi_weight_end: float = 0.8
    ce_weight_start: float = 0.8
    ce_weight_end: float = 0.6
    
    # Evaluation
    eval_lambda: float = 0.15  # Lambda value for evaluation
    
    @property
    def total_epochs(self) -> int:
        """Total number of training epochs"""
        return self.stage1_epochs + self.stage2_epochs
    
    def get_loss_weights(self, epoch: int) -> Tuple[float, float]:
        """Get FIDI and CE loss weights for current epoch"""
        progress = epoch / self.total_epochs
        
        # Progressive weight adjustment
        fidi_weight = self.fidi_weight_start + (
            self.fidi_weight_end - self.fidi_weight_start
        ) * progress
        
        ce_weight = self.ce_weight_start + (
            self.ce_weight_end - self.ce_weight_start
        ) * progress
        
        return fidi_weight, ce_weight
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.stage1_epochs > 0, "Stage 1 must have at least 1 epoch"
        assert self.stage2_epochs > 0, "Stage 2 must have at least 1 epoch"
        assert 0.0 <= self.semantic_weight <= 1.0, "Semantic weight must be between 0 and 1"
        assert self.semantic_embed_dim > 0, "Semantic embedding dimension must be positive"
        
        # Validate semantic part ranges
        assert 0.0 <= self.upper_body_range[0] < self.upper_body_range[1] <= 1.0
        assert self.upper_body_range[1] == self.lower_body_range[0]
        assert self.lower_body_range[1] == self.shoes_range[0]
        assert self.shoes_range[1] == 1.0