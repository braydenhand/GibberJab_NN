import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import time
import requests
import re
from collections import Counter
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

# Install required packages if not present
try:
    import tokenizers
except ImportError:
    print("Installing tokenizers package...")
    import subprocess
    subprocess.check_call(["pip", "install", "tokenizers"])
    import tokenizers

try:
    from transformers import PreTrainedTokenizerFast
except ImportError:
    print("Installing transformers package...")
    import subprocess
    subprocess.check_call(["pip", "install", "transformers"])
    from transformers import PreTrainedTokenizerFast

class BPETokenizer:
    """Byte Pair Encoding tokenizer using HuggingFace tokenizers library"""
    
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
    def train(self, corpus, verbose=True):
        """Train tokenizer on a text corpus"""
        if verbose:
            print("Training BPE tokenizer...")
            
        # Initialize a BPE tokenizer
        tokenizer = Tokenizer(models.BPE())
        
        # Configure pre-tokenization and post-processing
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        
        # Create a trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            min_frequency=2
        )
        
        # Prepare the corpus for training
        corpus_lines = corpus.split("\n")
        corpus_lines = [line for line in corpus_lines if len(line.strip()) > 0]
        
        # Train the tokenizer
        tokenizer.train_from_iterator(corpus_lines, trainer)
        
        # Wrap with PreTrainedTokenizerFast for easier use
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token="[UNK]"
        )
        
        # Set special token IDs
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        if verbose:
            print(f"BPE Tokenizer trained with vocabulary size: {len(self.tokenizer)}")
        
        return self
    
    def encode(self, text, max_length=None):
        """Convert text to token IDs"""
        # Add special tokens and pad/truncate to max_length
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length" if max_length else None,
            truncation=True if max_length else False,
            return_tensors="pt"
        )
        return encoded[0]  # Return tensor without batch dimension
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        # Filter out padding tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
            
        # Remove special tokens and decode
        token_ids = [id for id in token_ids if id not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def save(self, path):
        """Save tokenizer to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from file"""
        instance = cls()
        instance.tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        
        # Set special token IDs
        instance.pad_token_id = instance.tokenizer.pad_token_id
        instance.unk_token_id = instance.tokenizer.unk_token_id
        instance.bos_token_id = instance.tokenizer.bos_token_id
        instance.eos_token_id = instance.tokenizer.eos_token_id
        
        return instance

class TextDataset(Dataset):
    """Dataset for training the neural codec"""
    
    def __init__(self, corpus, tokenizer, sequence_length=64):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        
        # Split corpus into sentences/paragraphs
        sentences = []
        for para in corpus.split("\n"):
            para = para.strip()
            if not para:
                continue
                
            # Split long paragraphs into sentences
            para_sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in para_sentences:
                if len(sent.strip()) > 10:  # Only include reasonably long sentences
                    sentences.append(sent.strip())
        
        self.sentences = sentences
        print(f"Dataset created with {len(self.sentences)} sentences")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        text = self.sentences[idx]
        tokens = self.tokenizer.encode(text, self.sequence_length)
        return tokens

class SelfAttention(nn.Module):
    """Self-attention layer"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads, 
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Reshape for attention if needed (expected: seq_len, batch, hidden)
        orig_shape = x.shape
        if len(orig_shape) == 3:
            # [batch, seq, hidden] -> [seq, batch, hidden]
            x = x.permute(1, 0, 2)
            
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(x + attn_output)
        
        # Restore original shape if needed
        if len(orig_shape) == 3:
            # [seq, batch, hidden] -> [batch, seq, hidden]
            output = output.permute(1, 0, 2)
            
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1)]
        return x

class EnhancedEncoder(nn.Module):
    """Enhanced VAE Encoder with transformer architecture"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, num_heads=4, dropout=0.1):
        super(EnhancedEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Project embeddings to hidden dimension if needed
        self.embed_to_hidden = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()
        
        # Transformer layers
        self.attention1 = SelfAttention(hidden_dim, num_heads, dropout)
        self.attention2 = SelfAttention(hidden_dim, num_heads, dropout)
        
        # Projection to latent space
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Embed tokens and add positional encoding
        mask = (x != 0).float().unsqueeze(-1)  # Create mask for padding
        embedded = self.embedding(x) * mask
        embedded = self.pos_encoder(embedded)
        
        # Project to hidden dimension if needed
        hidden = self.embed_to_hidden(embedded)
        hidden = self.dropout(hidden)
        
        # Apply transformer layers
        hidden = self.attention1(hidden)
        hidden = self.attention2(hidden)
        
        # Global pooling across sequence dimension
        # [batch, seq, hidden] -> [batch, hidden, seq] -> [batch, hidden, 1] -> [batch, hidden]
        pooled = self.pooling(hidden.transpose(1, 2)).squeeze(-1)
        
        # Get mean and log variance for latent space
        mean = self.fc_mean(pooled)
        logvar = self.fc_logvar(pooled)
        
        # Sample from latent space
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z, mean, logvar

class EnhancedVectorQuantizer(nn.Module):
    """Enhanced Vector Quantization layer with commitment loss and EMA updates"""
    
    def __init__(self, latent_dim, num_embeddings=16, commitment_cost=0.25, decay=0.99, device="cpu"):
        super(EnhancedVectorQuantizer, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.device = device
        
        # Initialize codebook for each dimension
        self.codebooks = nn.ModuleList([
            nn.Embedding(num_embeddings, 1) for _ in range(latent_dim)
        ])
        
        # Initialize each codebook with uniformly spaced values
        for codebook in self.codebooks:
            values = torch.linspace(-1.5, 1.5, num_embeddings, device=device).unsqueeze(1)
            codebook.weight.data.copy_(values)
            
        # Register buffers for EMA updates
        self.register_buffer('_ema_cluster_size', torch.zeros(latent_dim, num_embeddings, device=device))
        self.register_buffer('_ema_w', torch.zeros(latent_dim, num_embeddings, 1, device=device))
        
        # Initialize the embeddings with uniform samples from N(-1, 1)
        for i, codebook in enumerate(self.codebooks):
            self._ema_w[i] = codebook.weight.data.clone()
    
    def forward(self, z, training=True):
        # z has shape [batch_size, latent_dim]
        batch_size = z.shape[0]
        
        # Quantize each dimension separately
        z_q = torch.zeros_like(z)
        indices = torch.zeros(batch_size, self.latent_dim, dtype=torch.long, device=self.device)
        
        # Compute the latent loss across all dimensions
        commitment_loss = 0.0
        
        for i in range(self.latent_dim):
            # Get the values for this dimension
            z_dim = z[:, i].unsqueeze(1)  # [batch_size, 1]
            
            # Calculate distances to codebook entries
            codebook = self.codebooks[i]
            distances = torch.sum((z_dim.unsqueeze(1) - codebook.weight) ** 2, dim=2)
            
            # Get closest codebook entry
            min_encodings = torch.zeros(batch_size, self.num_embeddings, device=self.device)
            min_encoding_indices = torch.argmin(distances, dim=1)
            indices[:, i] = min_encoding_indices
            
            # Create one-hot encodings
            min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
            
            # Get quantized values
            z_q_dim = torch.matmul(min_encodings, codebook.weight)
            z_q[:, i] = z_q_dim.squeeze()
            
            # Update the codebook if training with EMA
            if training:
                # Use EMA to update the embedding vectors
                with torch.no_grad():
                    # Cluster size
                    n = min_encodings.sum(0)
                    self._ema_cluster_size[i] = self._ema_cluster_size[i] * self.decay + (1 - self.decay) * n
                    
                    # Laplace smoothing
                    n_clipped = torch.max(n, torch.tensor(0.1, device=self.device))
                    
                    # Embed sum
                    embed_sum = torch.matmul(min_encodings.t(), z_dim)
                    self._ema_w[i] = self._ema_w[i] * self.decay + (1 - self.decay) * embed_sum
                    
                    # Update codebook weights
                    embed_normalized = self._ema_w[i] / n_clipped.unsqueeze(1)
                    codebook.weight.data.copy_(embed_normalized)
            
            # Compute commitment loss for this dimension
            commitment_loss += F.mse_loss(z_dim, z_q_dim.detach())
        
        # Compute codebook loss (encourages encodings to be close to codebook entries)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Use straight-through estimator for entire vector
        z_q_sg = z + (z_q - z).detach()
        
        return z_q_sg, vq_loss, indices

class EnhancedDecoder(nn.Module):
    """Enhanced VAE Decoder with transformer architecture"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, sequence_length,
                num_heads=4, dropout=0.1):
        super(EnhancedDecoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Transform latent vector to hidden sequence
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        
        # Create initial sequence from latent vector
        self.sequence_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * sequence_length),
            nn.LayerNorm(hidden_dim * sequence_length),
            nn.GELU(),
            nn.Linear(hidden_dim * sequence_length, hidden_dim * sequence_length),
            nn.LayerNorm(hidden_dim * sequence_length),
            nn.GELU()
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer layers
        self.attention1 = SelfAttention(hidden_dim, num_heads, dropout)
        self.attention2 = SelfAttention(hidden_dim, num_heads, dropout)
        self.attention3 = SelfAttention(hidden_dim, num_heads, dropout)
        
        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z):
        # Transform latent vector to hidden state
        hidden = self.latent_to_hidden(z)  # [batch, hidden]
        
        # Generate initial sequence
        hidden = self.sequence_generator(hidden)  # [batch, hidden * seq_len]
        hidden = hidden.view(-1, self.sequence_length, self.hidden_dim)  # [batch, seq_len, hidden]
        
        # Add positional encoding
        hidden = self.pos_encoder(hidden)
        hidden = self.dropout(hidden)
        
        # Apply transformer layers
        hidden = self.attention1(hidden)
        hidden = self.attention2(hidden)
        hidden = self.attention3(hidden)
        
        # Get token probabilities
        logits = self.output(hidden)
        
        return logits

class EnhancedNeuralTextCodec(nn.Module):
    """Enhanced neural network-based text codec optimized for FSK transmission"""
    
    def __init__(self, vocab_size=8000, embedding_dim=256, hidden_dim=512, 
                latent_dim=128, sequence_length=64, quantize=True, 
                codebook_size=16, device="cpu"):
        super(EnhancedNeuralTextCodec, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.codebook_size = codebook_size
        self.device = device
        self._step = 0  # For KL annealing
        
        # Calculate bits per dimension based on codebook size
        self.bits_per_dim = int(np.log2(codebook_size))
        print(f"Using {self.bits_per_dim} bits per dimension with latent_dim={latent_dim}")
        print(f"This allows for approximately {latent_dim * self.bits_per_dim / 8:.1f} bytes per message")
        
        # Enhanced encoder and decoder with transformer architecture
        self.encoder = EnhancedEncoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, 
            num_heads=8, dropout=0.1
        )
        
        self.decoder = EnhancedDecoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, sequence_length,
            num_heads=8, dropout=0.1
        )
        
        # Vector quantizer (if enabled)
        self.quantize = quantize
        if quantize:
            self.vector_quantizer = EnhancedVectorQuantizer(
                latent_dim, num_embeddings=codebook_size, commitment_cost=0.25, device=device
            )
        else:
            self.vector_quantizer = None
        
        # Tokenizer
        self.tokenizer = None
        
        # Move to device
        self.to(device)
    
    def forward(self, x, training=True):
        # Encode
        z, mean, logvar = self.encoder(x)
        
        # Quantize if enabled
        if self.quantize:
            z_q, vq_loss, encoding_indices = self.vector_quantizer(z, training)
        else:
            z_q = z
            vq_loss = 0
            encoding_indices = None
        
        # Decode
        logits = self.decoder(z_q)
        
        return logits, mean, logvar, vq_loss, encoding_indices
    
    def loss_function(self, x, logits, mean, logvar, vq_loss=0):
        # Create mask for padding
        mask = (x != 0).float()
        
        # Compute cross entropy loss only on non-padding tokens
        # Reshape for cross entropy
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = x.view(-1)
        
        # Compute cross entropy loss
        recon_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Reshape and apply mask
        recon_loss = recon_loss.view_as(x) * mask
        
        # Average over non-padding tokens
        recon_loss = recon_loss.sum() / mask.sum().clamp(min=1)
        
        # KL divergence with annealing schedule
        kl_weight = min(1.0, self._step / 10000.0) * 0.01  # Gradual increase to 0.01
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size
        
        # Total loss
        loss = recon_loss + kl_weight * kl_loss
        # Add a small term to encourage output diversity
        token_probs = F.softmax(logits, dim=-1)
        entropy = -(token_probs * torch.log(token_probs + 1e-10)).sum(dim=-1).mean()
        loss = loss - 0.01 * entropy  # Encourage diversity with small weight
        if self.quantize:
            loss = loss + vq_loss
        
        return loss, recon_loss, kl_loss, vq_loss
    
    def encode(self, text):
        """Encode text to compressed byte representation"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer first.")
        
        # Tokenize text
        tokens = self.tokenizer.encode(text, self.sequence_length).unsqueeze(0).to(self.device)
        
        # Encode to latent space
        with torch.no_grad():
            z, _, _ = self.encoder(tokens)
            
            # Quantize if enabled
            if self.quantize:
                _, _, encoding_indices = self.vector_quantizer(z, training=False)
                
                # Convert quantization indices to bytes
                indices = encoding_indices.cpu().numpy()[0]
                
                # Calculate how many indices we can pack per byte
                indices_per_byte = 8 // self.bits_per_dim
                
                # Debug info
                print(f"Quantized indices: shape={indices.shape}, range={indices.min()}-{indices.max()}")
                
                # Pack indices into bytes
                byte_data = bytearray()
                mask = (1 << self.bits_per_dim) - 1  # Bit mask for extracting values
                
                if indices_per_byte > 1:
                    # Pack multiple indices per byte
                    for i in range(0, len(indices), indices_per_byte):
                        byte_val = 0
                        for j in range(indices_per_byte):
                            if i + j < len(indices):
                                # Shift and add each index
                                byte_val |= (indices[i + j] & mask) << (j * self.bits_per_dim)
                        byte_data.append(byte_val)
                else:
                    # Each index needs multiple bytes
                    bytes_per_index = self.bits_per_dim // 8
                    for idx in indices:
                        for b in range(bytes_per_index):
                            # Extract each byte from the index
                            byte_val = (idx >> (b * 8)) & 0xFF
                            byte_data.append(byte_val)
                
                print(f"Packed {len(indices)} indices into {len(byte_data)} bytes")
                
                # Debug: print some of the byte values
                if len(byte_data) > 0:
                    preview = list(byte_data[:min(10, len(byte_data))])
                    print(f"First few bytes: {preview}")
                
                return bytes(byte_data)
            else:
                # Without quantization, use a simple scheme for floating point values
                z_np = z.cpu().numpy()[0]
                
                # Scale from roughly -3 to 3 (typical VAE values) to 0-255
                z_scaled = np.clip((z_np + 3) * 42.5, 0, 255).astype(np.uint8)
                
                print(f"Encoded {len(z_scaled)} float values to bytes")
                return bytes(z_scaled)
    
    def decode(self, byte_data):
        """Decode byte representation back to text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer first.")
        
        # Print debug info about the received data
        print(f"Decoding {len(byte_data)} bytes")
        if len(byte_data) > 0:
            preview = list(byte_data[:min(10, len(byte_data))])
            print(f"First few bytes: {preview}")
        
        # Convert bytes back to latent representation
        if self.quantize:
            # Calculate how many indices we can pack per byte
            indices_per_byte = 8 // self.bits_per_dim
            
            # Unpack bytes to indices
            indices = []
            mask = (1 << self.bits_per_dim) - 1  # Bit mask for extracting values
            
            if indices_per_byte > 1:
                # Multiple indices per byte
                for byte_val in byte_data:
                    for j in range(indices_per_byte):
                        # Extract each index from the byte
                        idx = (byte_val >> (j * self.bits_per_dim)) & mask
                        indices.append(idx)
            else:
                # Each index needs multiple bytes
                bytes_per_index = self.bits_per_dim // 8
                for i in range(0, len(byte_data), bytes_per_index):
                    if i + bytes_per_index <= len(byte_data):
                        # Combine multiple bytes into one index
                        idx = 0
                        for b in range(bytes_per_index):
                            idx |= byte_data[i + b] << (b * 8)
                        indices.append(idx)
            
            # Ensure indices list length matches latent_dim
            indices = indices[:self.latent_dim]
            if len(indices) < self.latent_dim:
                indices.extend([0] * (self.latent_dim - len(indices)))
            
            print(f"Unpacked {len(indices)} indices, range: {min(indices)}-{max(indices)}")
            
            # Convert indices to tensor
            indices_tensor = torch.tensor(indices, device=self.device).long().unsqueeze(0)
            
            # Reconstruct latent vector
            z = torch.zeros(1, self.latent_dim, device=self.device)
            
            for i in range(self.latent_dim):
                idx = indices_tensor[0, i]
                if idx < self.codebook_size:
                    embedding = self.vector_quantizer.codebooks[i](idx.unsqueeze(0))
                    z[0, i] = embedding.squeeze()
        
            # Debug: Print reconstructed latent vector stats
            print(f"Reconstructed latent vector shape: {z.shape}")
            print(f"Latent vector range: {z.min().item():.3f} to {z.max().item():.3f}")
        
        # Decode
        with torch.no_grad():
            logits = self.decoder(z)
            # Debug: Print logits information
            print(f"Decoder output shape: {logits.shape}")
            print(f"Logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")
            
            # Get top-k predictions for first few positions
            topk = torch.topk(logits[0, 0], k=5)
            print("Top 5 predictions for first position:")
            for prob, idx in zip(topk.values, topk.indices):
                token = self.tokenizer.decode([idx.item()])
                print(f"  {token}: {prob.item():.3f}")
            
            token_ids = torch.argmax(logits, dim=-1)[0]
            
            # Debug: Print token IDs
            print(f"Predicted token IDs shape: {token_ids.shape}")
            print(f"First few token IDs: {token_ids[:10].tolist()}")
        
        # Convert token IDs back to text
        decoded_text = self.tokenizer.decode(token_ids)
        print(f"Raw decoded text (before cleanup): {repr(decoded_text)}")
        
        return decoded_text
    
    def set_tokenizer(self, tokenizer):
        """Set the tokenizer to use for encoding/decoding"""
        self.tokenizer = tokenizer
        return self
    
    def train_model(self, train_dataset, val_dataset=None, batch_size=32, 
                   epochs=50, learning_rate=5e-4, beta1=0.9, beta2=0.999,
                   validate_every=1, checkpoint_every=1):
        """Train the model on a dataset with validation"""
        self._step = 0  # Track steps for KL annealing
        
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create validation dataloader if provided
        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            betas=(beta1, beta2),
            weight_decay=1e-5  # Reduced weight decay
        )
        
        # Learning rate scheduler with longer patience
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        print(f"Training model for {epochs} epochs...")
        
        best_loss = float('inf')
        patience = 0
        max_patience = 15  # Early stopping patience
        
        for epoch in range(epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Training
            self.train()
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            total_vq_loss = 0
            
            # Training progress bar
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in progress_bar:
                batch = batch.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits, mean, logvar, vq_loss, _ = self(batch, training=True)
                
                # Calculate loss
                loss, recon_loss, kl_loss, vq_loss_val = self.loss_function(batch, logits, mean, logvar, vq_loss)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Track losses
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                if vq_loss_val != 0:
                    total_vq_loss += vq_loss_val
                
                # Update progress bar with detailed losses
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{recon_loss.item():.4f}",
                    'kl': f"{kl_loss.item():.4f}",
                    'vq': f"{vq_loss_val if isinstance(vq_loss_val, float) else vq_loss_val.item():.4f}"
                })
                
                # Periodically print sample predictions
                if self._step % 100 == 0:
                    with torch.no_grad():
                        pred_tokens = torch.argmax(logits[0], dim=-1)
                        pred_text = self.tokenizer.decode(pred_tokens)
                        true_text = self.tokenizer.decode(batch[0])
                        print("\nSample prediction:")
                        print(f"True : {true_text[:100]}")
                        print(f"Pred : {pred_text[:100]}")
                
                self._step += 1
            
            # Calculate average losses
            avg_loss = total_loss / len(dataloader)
            avg_recon_loss = total_recon_loss / len(dataloader)
            avg_kl_loss = total_kl_loss / len(dataloader)
            avg_vq_loss = total_vq_loss / len(dataloader) if total_vq_loss > 0 else 0
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, VQ: {avg_vq_loss:.4f}")
            
            # Validation
            if val_dataloader is not None and (epoch + 1) % validate_every == 0:
                val_loss = self.validate(val_dataloader)
                # Update learning rate based on validation loss
                lr_scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                else:
                    patience += 1
                    if patience >= max_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            # Validate on sample text periodically
            if (epoch + 1) % validate_every == 0 or epoch == epochs - 1:
                self.validate_on_samples()
            
            # Save checkpoint after every epoch
            checkpoint_path = f"checkpoints/codec_epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        return self
    
    def validate(self, val_dataloader):
        """Validate model on validation data"""
        self.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(self.device)
                logits, mean, logvar, vq_loss, _ = self(batch, training=False)
                loss, _, _, _ = self.loss_function(batch, logits, mean, logvar, vq_loss)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate_on_samples(self):
        """Test compression/decompression on sample texts"""
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Neural network approaches to text compression can achieve better efficiency.",
            "This is a test of the neural codec system with transformer architecture."
        ]
        
        print("\nValidation on sample texts:")
        for i, text in enumerate(sample_texts):
            print(f"\nSample {i+1}: '{text}'")
            
            # Encode and decode
            encoded = self.encode(text)
            decoded = self.decode(encoded)
            
            print(f"Decoded: '{decoded}'")
            print(f"Bytes: {len(encoded)}")
            
            # Compute similarity score (simple word overlap)
            original_words = set(text.lower().split())
            decoded_words = set(decoded.lower().split())
            
            if len(original_words) > 0:
                overlap = len(original_words.intersection(decoded_words))
                similarity = overlap / len(original_words)
                print(f"Word similarity: {similarity:.2f}")
        
        print()
        
        return
    
    def save(self, path):
        """Save model to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save model parameters
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        
        # Save configuration
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "sequence_length": self.sequence_length,
            "quantize": self.quantize,
            "codebook_size": self.codebook_size
        }
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
    
    @classmethod
    def load(cls, path, device="cpu"):
        """Load model from disk"""
        # Load configuration
        with open(os.path.join(path, "config.pkl"), "rb") as f:
            config = pickle.load(f)
        
        # Create model with loaded configuration
        model = cls(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            sequence_length=config["sequence_length"],
            quantize=config["quantize"],
            codebook_size=config.get("codebook_size", 16),
            device=device
        )
        
        # Load model parameters
        model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=device))
        
        return model
    
    def compression_stats(self, text):
        """Calculate compression statistics for a text"""
        # Original size in bytes
        original_size = len(text.encode('utf-8'))
        
        # Compressed size
        compressed = self.encode(text)
        compressed_size = len(compressed)
        
        # Calculate ratio and savings
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        savings = (1 - 1/ratio) * 100 if ratio > 0 else 0
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': ratio,
            'space_saving': savings
        }

def create_deep_corpus(output_file="deep_corpus.txt", corpus_size=1000000):
    """Create a deep English corpus (simplified version)"""
    import requests
    
    # List of classic literature texts from Project Gutenberg (public domain)
    gutenberg_texts = [
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
        "https://www.gutenberg.org/files/98/98-0.txt",      # A Tale of Two Cities
        "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes
        "https://www.gutenberg.org/files/76/76-0.txt",      # Huckleberry Finn
        "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
        "https://www.gutenberg.org/files/345/345-0.txt",    # Dracula
    ]
    
    corpus_text = ""
    texts_used = 0
    
    for url in gutenberg_texts:
        try:
            print(f"Downloading {url}")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Extract text, removing Project Gutenberg header/footer
                text = response.text
                
                # Remove header (everything before the first empty line after "START")
                start_marker = "*** START OF"
                if start_marker in text:
                    start_pos = text.find(start_marker)
                    text = text[start_pos:]
                    para_break = text.find("\r\n\r\n")
                    if para_break > 0:
                        text = text[para_break+4:]
                
                # Remove footer (everything after "END")
                end_marker = "*** END OF"
                if end_marker in text:
                    end_pos = text.find(end_marker)
                    text = text[:end_pos]
                
                corpus_text += text + "\n\n"
                texts_used += 1
                
                if len(corpus_text) >= corpus_size:
                    break
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    # Save corpus to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(corpus_text)
    
    print(f"Created corpus with {len(corpus_text)} characters from {texts_used} texts")
    print(f"Saved to {output_file}")
    
    return output_file, corpus_text


def train_enhanced_neural_codec(corpus_path=None, output_path="enhanced_neural_codec", epochs=30, continue_training=False, corpus_size=5000000):
    """Train an enhanced neural codec on a corpus"""
    # Check for existing models
    if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "tokenizer")):
        print(f"Loading existing neural codec from {output_path}")
        
        # Load tokenizer
        tokenizer = BPETokenizer.load(os.path.join(output_path, "tokenizer"))
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=== Device Information ===")
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=======================\n")
        
        model = EnhancedNeuralTextCodec.load(output_path, device)
        model.set_tokenizer(tokenizer)
        
        # If continuing training, load the last checkpoint
        if continue_training:
            # Find the latest checkpoint
            checkpoint_files = [f for f in os.listdir("checkpoints") if f.startswith("codec_epoch_")]
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
                checkpoint_path = os.path.join("checkpoints", latest_checkpoint)
                print(f"Loading checkpoint: {checkpoint_path}")
                
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch']
                print(f"Resuming training from epoch {start_epoch}")
            else:
                print("No checkpoints found, starting from scratch")
                start_epoch = 0
        else:
            start_epoch = 0
        
        return model, start_epoch
    
    # Create or load corpus
    if corpus_path and os.path.exists(corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = f.read()
    else:
        _, corpus = create_deep_corpus(corpus_size=corpus_size)  # Use the provided corpus_size
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Device Information ===")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=======================\n")
    
    # Train BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=8000)
    tokenizer.train(corpus)
    
    # Split data into train and validation
    lines = [line for line in corpus.split("\n") if len(line.strip()) > 0]
    split_idx = int(len(lines) * 0.95)
    train_corpus = "\n".join(lines[:split_idx])
    val_corpus = "\n".join(lines[split_idx:])
    
    # Create datasets
    train_dataset = TextDataset(train_corpus, tokenizer, sequence_length=64)
    val_dataset = TextDataset(val_corpus, tokenizer, sequence_length=64)
    
    # Create model with enhanced architecture
    model = EnhancedNeuralTextCodec(
        vocab_size=len(tokenizer.tokenizer),
        embedding_dim=128,  # Reduced from 256
        hidden_dim=256,     # Reduced from 512
        latent_dim=64,      # Reduced from 128
        sequence_length=64,
        quantize=True,
        codebook_size=32,   # 5 bits per dimension
        device=device
    )
    
    model.set_tokenizer(tokenizer)
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Vocabulary size: {len(tokenizer.tokenizer)}")
    print(f"Embedding dimension: {128}")
    print(f"Hidden dimension: {256}")
    print(f"Latent dimension: {64}")
    print(f"Sequence length: {64}")
    print(f"Codebook size: {32}")
    
    # Train the model with smaller batch size
    model.train_model(
        train_dataset, 
        val_dataset=val_dataset,
        batch_size=32,  # Reduced from 64
        epochs=epochs,
        learning_rate=5e-4,
        validate_every=1,
        checkpoint_every=1
    )
    
    # Save model and tokenizer
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    tokenizer.save(os.path.join(output_path, "tokenizer"))
    
    print(f"Saved enhanced neural codec to {output_path}")
    
    return model, 0


def send_neural_encoded_message(message, codec, protocol_id=6):
    """Encode a message using neural codec and send via ggwave"""
    try:
        import ggwave
    except ImportError:
        print("ggwave module not found, installing stub for demonstration")
        
        # Create a simple stub for ggwave to demonstrate
        class GGWaveStub:
            @staticmethod
            def init():
                return None
                
            @staticmethod
            def encode(message, instance, protocol_id):
                print(f"[STUB] ggwave encoding message with protocol {protocol_id}")
                # Return dummy waveform
                return b'DUMMY_WAVEFORM'
                
            @staticmethod
            def free(instance):
                pass
        
        # Make our stub available as ggwave
        import sys
        sys.modules['ggwave'] = GGWaveStub
        import ggwave
    
    # Encode message
    print(f"Encoding message: '{message}'")
    encoded = codec.encode(message)
    
    # Print statistics
    stats = codec.compression_stats(message)
    print(f"Original message ({stats['original_size']} bytes): '{message}'")
    print(f"Neural encoded ({stats['compressed_size']} bytes), {stats['compression_ratio']:.2f}x compression")
    
    # Verify round-trip
    print("\nVerifying round-trip encoding/decoding:")
    decoded = codec.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    # Calculate similarity
    original_words = set(message.lower().split())
    decoded_words = set(decoded.lower().split())
    
    if len(original_words) > 0:
        overlap = len(original_words.intersection(decoded_words))
        similarity = overlap / len(original_words)
        print(f"Word similarity: {similarity:.2f}")
    
    try:
        # Initialize ggwave
        instance = ggwave.init()
        
        # Encode with ggwave - use the encoded bytes directly
        waveform = ggwave.encode(encoded, instance, protocol_id)
        
        # In a real implementation, this would play the audio
        print("Transmitting... (simulated for demo)")
        
        # Simulate play operation
        print("Transmission complete")
        
        # Free ggwave
        ggwave.free(instance)
        
    except Exception as e:
        print(f"Error: {e}")
    
    return encoded


if __name__ == "__main__":
    import shutil
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Neural Text Codec')
    parser.add_argument('--retrain', action='store_true', help='Force retraining even if model exists')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--corpus_size', type=int, default=5000000, help='Size of corpus in characters')
    parser.add_argument('--test_only', action='store_true', help='Skip training and only test the model')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from the last checkpoint')
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Set model path
    model_path = "enhanced_neural_codec"
    
    # Check if we should retrain from scratch
    if args.retrain and os.path.exists(model_path):
        print("Removing existing model for retraining...")
        shutil.rmtree(model_path)
    
    # Train or load neural codec
    if not args.test_only:
        # Create corpus if needed
        corpus_path = "deep_corpus.txt"
        if not os.path.exists(corpus_path) or args.retrain:
            create_deep_corpus(corpus_path, args.corpus_size)
        
        # Train the model
        model, start_epoch = train_enhanced_neural_codec(corpus_path, model_path, args.epochs, args.continue_training, args.corpus_size)
    else:
        # Load existing model
        if not os.path.exists(model_path):
            print("No existing model found. Please train a model first or remove --test_only flag.")
            exit(1)
        
        # Load tokenizer
        tokenizer = BPETokenizer.load(os.path.join(model_path, "tokenizer"))
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=== Device Information ===")
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("=======================\n")
        
        model = EnhancedNeuralTextCodec.load(model_path, device)
        model.set_tokenizer(tokenizer)
        print(f"Loaded model device: {next(model.parameters()).device}")
    
    # Example messages of different lengths
    short_message = "The quick brown fox jumps over the lazy dog."
    medium_message = "Neural network encoders can achieve better compression ratios than traditional methods by learning language patterns."
    long_message = """Neural network approaches to text compression can achieve much better efficiency than traditional methods by learning the statistical and semantic patterns in language. This allows for more compact representations while maintaining meaning."""
    
    # Test with the short message
    print("\n\n===== TESTING WITH SHORT MESSAGE =====")
    send_neural_encoded_message(short_message, model)
    
    # Test with the medium message
    print("\n\n===== TESTING WITH MEDIUM MESSAGE =====")
    send_neural_encoded_message(medium_message, model)
    
    # Test with the long message
    print("\n\n===== TESTING WITH LONG MESSAGE =====")
    send_neural_encoded_message(long_message, model)