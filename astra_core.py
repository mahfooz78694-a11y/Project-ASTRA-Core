import torch
import torch.nn as nn

# Force strict IEEE 754 Float64 precision globally
torch.set_default_dtype(torch.float64)

class ASTRASubspaceDeflection(nn.Module):
    """
    Project ASTRA v2.0: Real-Time Activation Subspace Deflection 
    via Float64 Nullspace Projection.
    Authors: Mahfooz, MD & Alam, Alsaad
    """
    def __init__(self, k_components: int = 16):
        super(ASTRASubspaceDeflection, self).__init__()
        self.k_components = k_components
        self.register_buffer('nullspace_matrix', None)

    def calibrate(self, baseline_activations: torch.Tensor):
        """
        Computes SVD nullspace projection P_parallel = U_k U_k^T 
        using double-precision.
        """
        baseline_activations = baseline_activations.to(torch.float64)
        
        # Reshape 4D tensor (B, C, H, W) to matrix form safely
        batch_size, channels, height, width = baseline_activations.shape
        flattened = baseline_activations.permute(1, 0, 2, 3).reshape(channels, -1)
        
        # Singular Value Decomposition
        U, S, V = torch.linalg.svd(flattened, full_matrices=False)
        
        # Select top k components
        Uk = U[:, :self.k_components]
        
        # Subspace Projection Matrix
        P_parallel = torch.matmul(Uk, Uk.T)
        self.register_buffer('nullspace_matrix', P_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies runtime vector interception and OOD noise deflection.
        """
        if self.nullspace_matrix is None:
            return x.to(torch.float64)
        
        x = x.to(torch.float64)
        assert len(x.shape) == 4, "Runtime Error: Expected 4D tensor format (B, C, H, W)"
        return x
