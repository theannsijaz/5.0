import torch
import torch.nn as nn
import copy

class TeacherStudentSOLIDER:
    """
    Teacher-Student framework for SOLIDER as described in the paper.
    The teacher network is a momentum-updated version of the student network.
    """
    def __init__(self, student_model, momentum=0.999):
        self.student = student_model
        # Create teacher as a copy of student
        self.teacher = copy.deepcopy(student_model)
        self.momentum = momentum
        
        # Disable gradient computation for teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def momentum_update(self):
        """
        Update teacher parameters using momentum update as described in the paper:
        θ_t = m * θ_t + (1 - m) * θ_s
        where θ_t are teacher parameters and θ_s are student parameters
        """
        for teacher_param, student_param in zip(self.teacher.parameters(), 
                                              self.student.parameters()):
            teacher_param.data = (
                self.momentum * teacher_param.data + 
                (1 - self.momentum) * student_param.data
            )
            
    def forward_student(self, images, lambda_val=0.5, return_semantic_loss=True, masked_features=None):
        """
        Forward pass through student network with optional masked feature input
        Args:
            images: Input images
            lambda_val: Semantic control parameter
            return_semantic_loss: Whether to return semantic supervision info
            masked_features: Optional masked features for prediction
        """
        # Always run the student on images. Masked features are used only to weight semantic loss,
        # not as image inputs to the backbone.
        return self.student(images, lambda_val, return_semantic_loss)
        
    @torch.no_grad()
    def forward_teacher(self, images, lambda_val=0.5):
        """
        Forward pass through teacher network (no gradient computation)
        Returns features and semantic supervision information
        """
        self.teacher.eval()
        features, logits, semantic_output = self.teacher(
            images, lambda_val, return_semantic_loss=True
        )
        return features, logits, semantic_output
    
    def get_student(self):
        """Get the student model"""
        return self.student
    
    def get_teacher(self):
        """Get the teacher model"""
        return self.teacher
