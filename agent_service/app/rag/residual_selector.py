import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Standard Residual Block with LayerNorm and GELU.
    Helps the model learn deep interactions without vanishing gradients.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualSelector(nn.Module):
    """
    The 'Model B' Architecture.
    Input: Query Embedding (1024), Document Embedding (1024)
    Output: Relevance Score (0-1)
    """
    def __init__(self, input_dim=1024, hidden_dim=512, num_blocks=2):
        super().__init__()
        # We project the 4 concatenated features: 
        # [Query, Doc, Hadamard(Q*D), Diff(|Q-D|)] -> 4 * 1024 = 4096 input features
        self.project = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.GELU()
        )
        
        # Deep interaction blocks
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        
        # Final classification head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, q, d):
        # Feature Engineering: Concatenate 4 distinct interaction types
        # 1. Raw Query (What user wants)
        # 2. Raw Doc (What we found)
        # 3. Hadamard (Matching keywords amplify each other)
        # 4. Diff (Missing info becomes prominent)
        feats = torch.cat([q, d, q * d, torch.abs(q - d)], dim=1)
        
        x = self.project(feats)
        
        for block in self.blocks:
            x = block(x)
            
        # Return scalar score
        return self.head(x).squeeze(-1)