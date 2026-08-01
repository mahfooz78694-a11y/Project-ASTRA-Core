import torch
import torch.nn as nn
import gc

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

    def _sanitize_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """
        Input Sanitization & Tensor Bounds Guard.
        Filters NaN, Inf, and clips uncalibrated extreme values.
        """
        t = t.to(torch.float64)
        t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
        t = torch.clamp(t, min=-1e6, max=1e6)
        return t

    def calibrate(self, baseline_activations: torch.Tensor):
        """
        Computes SVD nullspace projection P_perp = I - U_k U_k^T
        using double-precision.
        """
        baseline_activations = self._sanitize_tensor(baseline_activations)
        
        # Reshape 4D tensor (B, C, H, W) to matrix form safely
        batch_size, channels, height, width = baseline_activations.shape
        flattened = baseline_activations.permute(1, 0, 2, 3).reshape(channels, -1)
        
        # Singular Value Decomposition
        U, S, V = torch.linalg.svd(flattened, full_matrices=False)
        
        # Select top k components
        Uk = U[:, :self.k_components]
        
        # Identity matrix
        I_matrix = torch.eye(channels, dtype=torch.float64, device=baseline_activations.device)

        # Subspace Projection Matrix P_perp
        P_perp = I_matrix - torch.matmul(Uk, Uk.T)
        self.register_buffer('nullspace_matrix', P_perp)

        # Memory & Tensor Cleanup
        del flattened
        del U
        del S
        del V
        del Uk
        del I_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies runtime vector interception and OOD noise deflection.
        """
        x = self._sanitize_tensor(x)
        assert len(x.shape) == 4, "Runtime Error: Expected 4D tensor format (B, C, H, W)"

        if self.nullspace_matrix is None:
            return x

        batch_size, channels, height, width = x.shape
        
        # Flatten for matrix multiplication
        flattened = x.permute(1, 0, 2, 3).reshape(channels, -1)

        # Deflect adversarial noise using P_perp
        projected = torch.matmul(self.nullspace_matrix, flattened)

        # Reshape back to original 4D shape
        projected_4d = projected.reshape(channels, batch_size, height, width).permute(1, 0, 2, 3)

        # Memory & Tensor Cleanup
        del flattened
        del projected

        return projected_4d
