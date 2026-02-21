import torch
# 1. IMPORTANT: Import components so the @register decorators actually run!
import components 
from core.config import TransformerConfig
from model.transformer import Transformer

# 2. Define your dream model in the config
config = TransformerConfig.gpt_nano() 

# 3. Instantiate the model
model = Transformer(config)

# 4. Create dummy data (Batch size 2, Sequence length 8)
dummy_input = torch.randint(0, config.vocab_size, (2, 8))

# 5. Run it!
logits = model(dummy_input)

print(f"Success! Input shape: {dummy_input.shape}")
print(f"Output (Logits) shape: {logits.shape}") # Should be (2, 8, vocab_size)
print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")