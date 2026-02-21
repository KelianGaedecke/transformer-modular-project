import torch 
import torch.nn as nn
from core.interfaces import FFNStrategy
from core.registry import register
from core.config import TransformerConfig

@register('ffn','standard')
class StandardFFN(FFNStrategy):
    def __init__(self, config: TransformerConfig):
        super().__init__() # run init logic from parent class 

        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(), 
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)