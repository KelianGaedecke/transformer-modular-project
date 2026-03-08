# components/__init__.py
# All imports here are side-effect-only: they trigger @register decorators.
from .attention import MultiHeadAttention, MultiScaleAttention
from .positional import NonePositionalEmbedding, Fourier3DPositionalEmbedding, DistanceBiasPositionalEmbedding
from .ffn import StandardFFN, SwiGLUFFN
from .message_passing import SchNetMessagePassing
