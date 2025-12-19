import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Learns which dimensions of the 1024 embedding to trust/suppress.
    """
    def __init__(self, input_dim=1024, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction, bias=False)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze (Global Information)
        # Note: Since input is already a 1D vector per batch, we just project down
        y = self.relu(self.fc1(x))
        # Excitation (Feature Importance)
        y = self.sigmoid(self.fc2(y))
        # Scale input features
        return x * y

class DeepResidualExpander(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super().__init__()
        
        # 1. Feature Attention (The "Filter")
        # Learns to suppress noise words immediately
        self.se_block = SEBlock(input_dim)
        
        # 2. Deep Shared Encoder (Residual MLP)
        # Two residual blocks for deeper logic reasoning
        self.shared_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.shared_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 3. Task-Specific Heads
        # We start these at Zero-Init so they learn Delta shifts
        self.head_para = nn.Linear(hidden_dim, input_dim)
        self.head_broad = nn.Linear(hidden_dim, input_dim)
        self.head_tech = nn.Linear(hidden_dim, input_dim)
        self.head_expl = nn.Linear(hidden_dim, input_dim)

        self._init_zero_heads()

    def _init_zero_heads(self):
        for head in [self.head_para, self.head_broad, self.head_tech, self.head_expl]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        # A. Apply Attention Filter
        filtered_x = self.se_block(x)
        
        # B. Shared Processing (with Residuals if dims matched, but here we change dims)
        # So we just do standard deep encoding
        h1 = self.shared_1(filtered_x)
        h2 = self.shared_2(h1) + h1 # Residual connection in latent space
        
        # C. Generate Deltas
        d_para = self.head_para(h2)
        d_broad = self.head_broad(h2)
        d_tech = self.head_tech(h2)
        d_expl = self.head_expl(h2)
        
        # D. Add to ORIGINAL input (Residual Learning)
        # Output = Input + Delta
        return (
            F.normalize(x + d_para, p=2, dim=1),
            F.normalize(x + d_broad, p=2, dim=1),
            F.normalize(x + d_tech, p=2, dim=1),
            F.normalize(x + d_expl, p=2, dim=1)
        )