import torch 
import torch.nn as nn
from core.interfaces import FFNStrategy
from core.registry import register
from core.config import PointCloudTransformerConfig

@register('ffn','standard')
class StandardFFN(FFNStrategy):
    def __init__(self, config: PointCloudTransformerConfig):
        super().__init__() # run init logic from parent class 

        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(), 
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register('ffn', 'swiglu')
class SwiGLUFFN(FFNStrategy):
    """SwiGLU feed-forward (used in LLaMA, PaLM, many modern models).
    Gate mechanism: output = (W1·x) * silu(W2·x), then projected back.
    d_ff is split across the two parallel projections."""

    def __init__(self, config: PointCloudTransformerConfig):
        super().__init__()
        self.w1  = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w2  = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.out = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.out(self.w1(x) * torch.nn.functional.silu(self.w2(x))))