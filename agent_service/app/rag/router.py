import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralRouterFusion(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        
        # 1. The "Manager" Network
        # It takes the Query vector and decides which Expert is needed.
        self.router = nn.Sequential(
            nn.Linear(input_dim, 128),  # Compress semantic meaning
            nn.LayerNorm(128),          # Stability
            nn.ReLU(),
            nn.Linear(128, 4),          # Output 4 weights (Para, Broad, Tech, Expl)
            nn.Sigmoid()                # Scale between 0.0 and 1.0
        )
        
        # Base weight is still best kept static or learned globally, 
        # as we almost always want to trust the backbone.
        self.w_base = nn.Parameter(torch.tensor(1.0))
        
        # A global scaler to allow the router to output values > 1.0 if needed
        self.global_scale = nn.Parameter(torch.tensor(2.0)) 

    def forward(self, q_emb, s_base, d_para, d_broad, d_tech, d_expl):
        """
        q_emb:  [Batch, 1024] - The query embedding
        s_base: [Batch]       - Original scores
        d_...:  [Batch]       - Deltas for each head
        """
        # 1. Generate Dynamic Weights for this batch
        # shape: [Batch, 4]
        # We multiply by global_scale so the range is [0, 2.0] instead of [0, 1.0]
        dynamic_weights = self.router(q_emb) * self.global_scale
        
        # 2. Extract individual dynamic weights (per sample)
        w_p = dynamic_weights[:, 0]
        w_b = dynamic_weights[:, 1]
        w_t = dynamic_weights[:, 2]
        w_e = dynamic_weights[:, 3]
        
        # 3. Fuse
        # Note: All operations are element-wise [Batch]
        final_score = (
            (self.w_base * s_base) +
            (w_p * d_para) +
            (w_b * d_broad) +
            (w_t * d_tech) +
            (w_e * d_expl)
        )
        
        return final_score, dynamic_weights