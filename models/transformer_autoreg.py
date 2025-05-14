import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()

        assert n_embd % n_head == 0, "embedding dimension is not divisible by number of heads"
        
        # Key, query, and value
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        
        # Dropout
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        
        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Save parameters
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Register buffer for mask (no parameters)
        self.register_buffer("mask", None)
        
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dimensionality
        
        # Calculate query, key, and value for all heads in the current batch
        # (B, T, C) -> (B, T, n_head, C/n_head) -> (B, n_head, T, C/n_head)
        head_size = self.n_embd // self.n_head
        k = self.key(x).view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # Self-attention: (B, nh, T, T)
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Create a mask if it doesn't exist
        if self.mask is None or self.mask.shape[0] != T:
            self.mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        
        # Apply the mask (sets upper triangular part to -inf)
        att = att.masked_fill(self.mask, float('-inf'))
        
        # Attention weights via softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Apply attention weights to values
        out = att @ v  # (B, nh, T, hs)
        
        # Reshaping: (B, nh, T, hs) -> (B, T, n_head, hs) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection and dropout
        out = self.resid_drop(self.proj(out))
        return out


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1, expansion_factor=4):
        super().__init__()

        # Original embeddings to a larger layer
        self.c_fc = nn.Linear(n_embd, expansion_factor * n_embd)
        # GELU activation
        self.gelu = nn.GELU()
        # Back to the original embedding dimensions
        self.c_proj = nn.Linear(expansion_factor * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ByteTransformer(nn.Module):

    def __init__(
        self, 
        vocab_size=256,    # 256 byte values
        block_size=256,    # Maximum sequence length
        n_embd=512,        # Embedding dimension
        n_head=8,          # Number of attention heads
        n_layer=8,         # Number of transformer blocks
        dropout=0.1,       # Dropout
    ):
        super().__init__()
        
        # Parameters
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        
        # Embed (and learn) position representation
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        
        # n_layer transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        
        # Final layer norm and output
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        # Batch size and sequence length
        B, T = idx.size() 
        assert T <= self.block_size, f"Sequence is too long ({T}), block size is only {self.block_size}"
        
        # (B, T, n_embd)
        token_embeddings = self.tok_emb(idx) 
        
        # Add positional embeddings (slice to current sequence length)
        position_embeddings = self.pos_emb[:, :T, :]  # (1, T, n_embd)
        x = token_embeddings + position_embeddings  # (B, T, n_embd)
        
        # Go through transformer blocks
        x = self.transformer_blocks(x)  # (B, T, n_embd)
        
        # Final layer norm
        x = self.ln_f(x)  # (B, T, n_embd)
        
        # Project to logits
        logits = self.head(x)  # (B, T, vocab_size)
        
        # If there are targets are provided, compute loss
        loss = None
        if targets is not None:
            # Reshape for cross entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),  # (B*T, vocab_size)
                targets.view(-1),                  # (B*T)
            )
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # If sequence exceeds block size, truncate it
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass through the model
            logits, _ = self.forward(idx_cond)
            
            # Get logits for the last position only
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # Sample from the distribution
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append sampled token to the sequence
            idx = torch.cat((idx, next_idx), dim=1)  # (B, seq_len+1)
            
        return idx 