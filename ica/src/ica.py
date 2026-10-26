import torch
import torch.nn as nn

class IntentConfidenceArbitration(nn.Module):
    def __init__(self, eeg_dim=1024, llm_dim=1536, hidden_dim=512):
        super().__init__()
        self.eeg_projector = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.text_projector = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.arbitrator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, eeg_features, llm_last_hidden_state):
        eeg_summary = torch.mean(eeg_features, dim=1)
        text_summary = llm_last_hidden_state[:, -1, :]
        
        eeg_emb = self.eeg_projector(eeg_summary)
        text_emb = self.text_projector(text_summary)
        
        combined = torch.cat([eeg_emb, text_emb], dim=1)
        return self.arbitrator(combined)