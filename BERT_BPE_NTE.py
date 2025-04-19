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

# Install required packages if not present
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers package...")
    import subprocess
    subprocess.check_call(["pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
except ImportError:
    print("Installing tokenizers package...")
    import subprocess
    subprocess.check_call(["pip", "install", "tokenizers"])
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

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
    
    def __init__(self, corpus, tokenizer, sequence_length=64, sentence_encoder=None):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.sentence_encoder = sentence_encoder
        
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
        
        # If sentence encoder is available, also return the sentence embedding
        if self.sentence_encoder is not None:
            with torch.no_grad():
                sentence_embedding = self.sentence_encoder.encode(text, convert_to_tensor=True)
            return tokens, sentence_embedding
        
        return tokens

# Modified SentenceBERTEncoder class that integrates with the pretrained encoder
class SentenceBERTEncoder(nn.Module):
    """Enhanced VAE Encoder with Sentence-BERT integration"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, 
                 sbert_dim=768, num_heads=4, dropout=0.1, freeze_sbert=True):
        super(SentenceBERTEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Project embeddings to hidden dimension if needed
        self.embed_to_hidden = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else nn.Identity()
        
        # Add a projection layer from SBERT embeddings to hidden dimension
        # Default SBERT embedding dimension is usually 768
        self.sbert_to_hidden = nn.Sequential(
            nn.Linear(sbert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Fusion mechanism to combine token-based and SBERT representations
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Transformer layers
        self.attention1 = SelfAttention(hidden_dim, num_heads, dropout)
        self.attention2 = SelfAttention(hidden_dim, num_heads, dropout)
        
        # Projection to latent space
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Flag to determine if we're using SBERT embeddings
        self.use_sbert = True
    
    def forward(self, x, sbert_embedding=None):
        if x.dtype != torch.long:
            x = x.long()
        
        # Print input shape for debugging
        print(f"Input shape: {x.shape}")
        
        # Check input dimensions and reshape if needed
        if len(x.shape) == 4:
            # We have a 4D tensor [batch, dim1, dim2, features]
            # Reshape to 3D by flattening the middle dimensions
            batch_size, dim1, dim2, _ = x.shape
            x = x.reshape(batch_size, dim1 * dim2, -1)
            print(f"Reshaped input to: {x.shape}")
        
        # Embed tokens and add positional encoding
        mask = (x != 0).float().unsqueeze(-1)  # Create mask for padding
        embedded = self.embedding(x) * mask
        embedded = self.pos_encoder(embedded)
        
        # Project to hidden dimension if needed
        hidden = self.embed_to_hidden(embedded)
        hidden = self.dropout(hidden)
        
        print(f"Hidden shape after embedding: {hidden.shape}")
        
        # If SBERT embedding is provided, integrate it
        if sbert_embedding is not None and self.use_sbert:
            # Project SBERT embedding to hidden dimension
            sbert_hidden = self.sbert_to_hidden(sbert_embedding).unsqueeze(1)
            print(f"SBERT hidden shape: {sbert_hidden.shape}")
            
            # Ensure dimensions are compatible
            if len(hidden.shape) != 3:
                # Reshape to 3D tensor
                if len(hidden.shape) == 4:
                    batch_size, dim1, dim2, features = hidden.shape
                    hidden = hidden.reshape(batch_size, dim1 * dim2, features)
                    print(f"Reshaped hidden to: {hidden.shape}")
                else:
                    raise ValueError(f"Cannot handle hidden shape: {hidden.shape}")
            
            # Get dimensions for expansion
            batch_size, seq_len, _ = hidden.shape
            
            # Expand SBERT hidden to match sequence length
            sbert_hidden = sbert_hidden.expand(-1, seq_len, -1)
            print(f"Expanded SBERT hidden shape: {sbert_hidden.shape}")
            
            # Compute fusion gate
            fusion_input = torch.cat([hidden, sbert_hidden], dim=-1)
            gate = self.fusion_gate(fusion_input)
            
            # Fuse representations
            hidden = gate * hidden + (1 - gate) * sbert_hidden
        
        # Apply transformer layers
        hidden = self.attention1(hidden)
        hidden = self.attention2(hidden)
        
        # Global pooling across sequence dimension
        pooled = self.pooling(hidden.transpose(1, 2)).squeeze(-1)
        
        # Get mean and log variance for latent space
        mean = self.fc_mean(pooled)
        logvar = self.fc_logvar(pooled)
        
        # Sample from latent space
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z, mean, logvar

class SelfAttention(nn.Module):
    """Self-attention layer"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True  # Set to True to avoid permute operations
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Now x is expected in [batch, seq, hidden] format due to batch_first=True
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(x + attn_output)
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term with correct dimension
        div_term_size = d_model // 2
        div_term = torch.exp(torch.arange(0, div_term_size).float() * (-np.log(10000.0) / d_model))
        
        # Ensure indices don't go out of bounds for even/odd d_model
        pe[:, 0::2] = torch.sin(position * div_term[:d_model//2 + d_model%2])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        self.d_model = d_model
        
    def forward(self, x):
        # Check input dimensions and handle appropriately
        if len(x.size()) != 3:
            # Handle unexpected input shape
            print(f"Warning: Expected 3D input (batch, seq, features), got shape {x.shape}")
            
            # Try to infer dimensions
            if len(x.size()) == 2:
                # Assume (batch, seq) and add feature dimension
                x = x.unsqueeze(-1)
                print(f"Reshaped to {x.shape}")
            else:
                # Return unchanged if we can't handle it
                return x
        
        # Get input dimensions safely
        batch_size, seq_len, emb_dim = x.size()
        
        # If dimensions don't match, create compatible positional encoding
        if emb_dim != self.d_model:
            device = x.device
            # Create new positional encoding matching the input embedding dimension
            temp_pe = torch.zeros(1, seq_len, emb_dim, device=device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
            
            # Handle the special case where emb_dim is 1
            if emb_dim == 1:
                # Simple sinusoidal encoding for single dimension
                temp_pe[:, :, 0] = torch.sin(position * 0.01)
                return x + temp_pe
            
            div_term_size = max(1, emb_dim // 2)  # Ensure at least size 1
            div_term = torch.exp(torch.arange(0, div_term_size, device=device).float() * 
                                (-np.log(10000.0) / max(2, emb_dim)))
            
            # Safely fill the positional encoding
            if emb_dim >= 2:
                # Handle even/odd dimensions safely
                sin_idx = min(emb_dim//2 + emb_dim%2, emb_dim)
                cos_idx = min(emb_dim//2, emb_dim - 1)
                
                # Use slicing to avoid index errors
                sin_term = div_term[:sin_idx]
                cos_term = div_term[:cos_idx]
                
                # Apply positional encoding
                temp_pe[:, :, 0:emb_dim:2] = torch.sin(position * sin_term.unsqueeze(0))
                if emb_dim > 1:  # Only add cosine terms if we have more than 1 dimension
                    temp_pe[:, :, 1:emb_dim:2] = torch.cos(position * cos_term.unsqueeze(0))
            
            return x + temp_pe
        else:
            # Standard case - use precomputed PE
            return x + self.pe[:, :seq_len]

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

# Enhanced decoder that can take advantage of conditioning on SBERT embeddings
class EnhancedDecoder(nn.Module):
    """Enhanced VAE Decoder with transformer architecture and SBERT conditioning"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, sequence_length,
                 sbert_dim=768, num_heads=4, dropout=0.1, use_sbert_conditioning=True):
        super(EnhancedDecoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.use_sbert_conditioning = use_sbert_conditioning
        
        # Transform latent vector to hidden sequence
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        
        # Conditional projection from SBERT embeddings
        if use_sbert_conditioning:
            self.sbert_to_cond = nn.Sequential(
                nn.Linear(sbert_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            
            # Gate for conditioning
            self.cond_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
        # Create initial sequence from latent vector
        self.sequence_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * sequence_length),
            nn.LayerNorm(hidden_dim * sequence_length),
            nn.GELU()
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer layers (using batch_first=True for efficiency)
        self.attention_layers = nn.ModuleList([
            SelfAttention(hidden_dim, num_heads, dropout) for _ in range(3)
        ])
        
        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z, sbert_embedding=None):
        # Transform latent vector to hidden state
        hidden = self.latent_to_hidden(z)  # [batch, hidden]
        
        # Apply SBERT conditioning if available
        if sbert_embedding is not None and self.use_sbert_conditioning:
            sbert_cond = self.sbert_to_cond(sbert_embedding)
            
            # Compute conditioning gate
            gate_input = torch.cat([hidden, sbert_cond], dim=-1)
            gate = self.cond_gate(gate_input)
            
            # Apply gated conditioning
            hidden = gate * hidden + (1 - gate) * sbert_cond
        
        # Generate initial sequence
        hidden = self.sequence_generator(hidden)  # [batch, hidden * seq_len]
        hidden = hidden.view(-1, self.sequence_length, self.hidden_dim)  # [batch, seq_len, hidden]
        
        # Add positional encoding
        hidden = self.pos_encoder(hidden)
        hidden = self.dropout(hidden)
        
        # Apply transformer layers
        for attention_layer in self.attention_layers:
            hidden = attention_layer(hidden)
        
        # Get token probabilities
        logits = self.output(hidden)
        
        return logits

class EnhancedNeuralTextCodec(nn.Module):
    """Enhanced neural network-based text codec with Sentence-BERT integration"""
    
    def __init__(self, vocab_size=8000, embedding_dim=128, hidden_dim=128, 
                 latent_dim=64, sequence_length=64, quantize=True, 
                 codebook_size=16, device="cpu", sbert_model_name="all-MiniLM-L6-v2"):
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
        
        # Load sentence-BERT model
        print(f"Loading Sentence-BERT model: {sbert_model_name}")
        self.sentence_encoder = SentenceTransformer(sbert_model_name)
        
        # Freeze the SBERT model for efficiency
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False
            
        # Get SBERT embedding dimension
        self.sbert_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        print(f"Sentence-BERT embedding dimension: {self.sbert_dim}")
        
        # Enhanced encoder with SBERT integration
        self.encoder = SentenceBERTEncoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, 
            sbert_dim=self.sbert_dim, num_heads=4, dropout=0.1
        )
        
        # Enhanced decoder with SBERT conditioning
        self.decoder = EnhancedDecoder(
            vocab_size, embedding_dim, hidden_dim, latent_dim, sequence_length,
            sbert_dim=self.sbert_dim, num_heads=4, dropout=0.1
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
        
        # Move SBERT model to the same device
        self.sentence_encoder = self.sentence_encoder.to(device)
    
    def forward(self, x, sbert_embedding=None, training=True):
        # Compute SBERT embedding if not provided
        if sbert_embedding is None and training:
            # We need to decode the tokens back to text to compute SBERT embedding
            # This is expensive, so should be done in the dataset instead when possible
            batch_texts = []
            for tokens in x:
                # Handle different tensor formats
                if isinstance(tokens, torch.Tensor):
                    # Convert tensor to list of integers
                    token_list = tokens.cpu().tolist()
                    
                    # Make sure we have integers, not nested lists
                    if isinstance(token_list, list) and any(isinstance(item, list) for item in token_list):
                        # We have a 2D tensor, flatten it
                        flat_tokens = []
                        for sublist in token_list:
                            if isinstance(sublist, list):
                                flat_tokens.extend([t for t in sublist if t != 0])  # Skip padding tokens (0)
                            else:
                                if sublist != 0:  # Skip padding token
                                    flat_tokens.append(sublist)
                        token_list = flat_tokens
                else:
                    # Already a list
                    token_list = tokens
                    
                # Skip any padding tokens (usually 0)
                token_list = [t for t in token_list if t != 0]
                
                try:
                    # Try to decode
                    text = self.tokenizer.decode(token_list, skip_special_tokens=True)
                    batch_texts.append(text)
                except TypeError as e:
                    # Fall back to character-by-character decoding
                    parts = []
                    for token_id in token_list:
                        try:
                            parts.append(self.tokenizer.decode([token_id], skip_special_tokens=True))
                        except:
                            pass  # Skip problematic tokens
                    text = "".join(parts)
                    batch_texts.append(text)
            with torch.no_grad():
                sbert_embedding = self.sentence_encoder.encode(batch_texts, convert_to_tensor=True)
        
        # Encode
        z, mean, logvar = self.encoder(x, sbert_embedding)
        
        # Quantize if enabled
        if self.quantize:
            z_q, vq_loss, encoding_indices = self.vector_quantizer(z, training)
        else:
            z_q = z
            vq_loss = 0
            encoding_indices = None
        
        # Decode (with SBERT conditioning)
        logits = self.decoder(z_q, sbert_embedding)
        
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
        
        if self.quantize:
            loss = loss + vq_loss
        
        return loss, recon_loss, kl_loss, vq_loss
    
    def encode(self, text):
        """Encode text to compressed byte representation"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer first.")
        
        # Tokenize text
        tokens = self.tokenizer.encode(text, self.sequence_length).unsqueeze(0).to(self.device)
        
        # Get SBERT embedding
        with torch.no_grad():
            sbert_embedding = self.sentence_encoder.encode(text, convert_to_tensor=True).to(self.device)
            sbert_embedding = sbert_embedding.unsqueeze(0)  # Add batch dimension
        
        # Encode to latent space
        with torch.no_grad():
            z, _, _ = self.encoder(tokens, sbert_embedding)
            
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
                
                # Store SBERT embedding for later use in decoding
                self._last_sbert_embedding = sbert_embedding
                
                return bytes(byte_data)
            else:
                # Without quantization, use a simple scheme for floating point values
                z_np = z.cpu().numpy()[0]
                
                # Scale from roughly -3 to 3 (typical VAE values) to 0-255
                z_scaled = np.clip((z_np + 3) * 42.5, 0, 255).astype(np.uint8)
                
                # Store SBERT embedding for later use in decoding
                self._last_sbert_embedding = sbert_embedding
                
                print(f"Encoded {len(z_scaled)} float values to bytes")
                return bytes(z_scaled)
    
    def decode(self, byte_data, use_stored_sbert=True):
        """Decode byte representation back to text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer first.")
        
        # Print debug info about the received data
        print(f"Decoding {len(byte_data)} bytes")
        
        # Get SBERT embedding if available and requested
        sbert_embedding = self._last_sbert_embedding if hasattr(self, '_last_sbert_embedding') and use_stored_sbert else None
        
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
        
        # Decode with SBERT conditioning if available
        with torch.no_grad():
            logits = self.decoder(z, sbert_embedding)
            # Fix: Convert tensor to list of integers
            token_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()
            
            # Make sure we have a flat list of integers
            if isinstance(token_ids, list) and any(isinstance(item, list) for item in token_ids):
                # If we have a list of lists, flatten it
                flat_token_ids = []
                for sublist in token_ids:
                    if isinstance(sublist, list):
                        flat_token_ids.extend(sublist)
                    else:
                        flat_token_ids.append(sublist)
                token_ids = flat_token_ids
        
        # Convert token IDs back to text
        try:
            # Add debug print to see what we're sending to the tokenizer
            print(f"Token IDs type: {type(token_ids)}, example: {token_ids[:5] if len(token_ids) > 5 else token_ids}")
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            return decoded_text
        except TypeError as e:
            # If we still get an error, provide more debugging info
            print(f"Error decoding tokens: {e}")
            print(f"Token IDs: {token_ids}")
            
            # Try an alternative approach
            if hasattr(self.tokenizer, 'convert_ids_to_tokens'):
                # First convert ids to tokens, then join them
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
                decoded_text = self.tokenizer.convert_tokens_to_string(tokens)
                return decoded_text
            else:
                # Last resort: decode one by one
                decoded_parts = []
                for token_id in token_ids:
                    try:
                        decoded_parts.append(self.tokenizer.decode([token_id], skip_special_tokens=True))
                    except:
                        pass  # Skip problematic tokens
                return " ".join(decoded_parts)
    
    def set_tokenizer(self, tokenizer):
        """Set the tokenizer to use for encoding/decoding"""
        self.tokenizer = tokenizer
        return self
    
    def train_model(self, train_dataset, val_dataset=None, batch_size=32, 
                epochs=50, learning_rate=5e-4, beta1=0.9, beta2=0.999,
                validate_every=1, checkpoint_every=1):
        """Train the model on a dataset with validation"""
        self._step = 0  # Track steps for KL annealing
        
        # Configure dataloaders
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            pin_memory=True, num_workers=0)
        
        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    pin_memory=True, num_workers=0)
        
        # Optimizer setup
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            betas=(beta1, beta2),
            weight_decay=1e-5
        )
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Gradient scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        print(f"Training model for {epochs} epochs...")
        
        best_loss = float('inf')
        patience = 0
        max_patience = 10  # Early stopping patience
        
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
            
            for batch_data in progress_bar:
                # Process batch data
                sbert_embeddings = None
                
                # Handle tuple case (tokens, sbert_embeddings)
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    batch, sbert_embeddings = batch_data
                    
                    # Convert sbert_embeddings to tensor if needed
                    if isinstance(sbert_embeddings, list):
                        sbert_embeddings = torch.tensor(sbert_embeddings, dtype=torch.float32)
                else:
                    batch = batch_data
                
                # Handle batch processing
                if isinstance(batch, list):
                    # If batch is a list of tensors with potentially different sizes
                    # We need to make sure they have consistent dimensions
                    
                    # Make sure all items are tensors
                    batch_tensors = []
                    for item in batch:
                        if not isinstance(item, torch.Tensor):
                            item = torch.tensor(item)
                        batch_tensors.append(item)
                    
                    # Check if all tensors have the same shape
                    shapes = [tensor.shape for tensor in batch_tensors]
                    if len(set(str(shape) for shape in shapes)) > 1:
                        # Tensors have different shapes, need to pad or resize
                        
                        # Check if this is a 2D tensor issue
                        if all(len(tensor.shape) == 2 for tensor in batch_tensors):
                            # Get the maximum dimensions across all tensors
                            max_dim0 = max(tensor.shape[0] for tensor in batch_tensors)
                            max_dim1 = max(tensor.shape[1] for tensor in batch_tensors)
                            
                            # Pad all tensors to the maximum dimensions
                            padded_tensors = []
                            for tensor in batch_tensors:
                                # Create new padded tensor
                                padded = torch.zeros((max_dim0, max_dim1), dtype=tensor.dtype)
                                # Copy original tensor values
                                padded[:tensor.shape[0], :tensor.shape[1]] = tensor
                                padded_tensors.append(padded)
                            
                            batch = torch.stack(padded_tensors)
                        else:
                            # For 1D tensors or other cases
                            max_len = max(len(tensor) for tensor in batch_tensors)
                            padded_tensors = []
                            for tensor in batch_tensors:
                                if len(tensor) < max_len:
                                    padding = torch.zeros(max_len - len(tensor), dtype=tensor.dtype)
                                    padded = torch.cat([tensor, padding])
                                else:
                                    padded = tensor
                                padded_tensors.append(padded)
                            
                            batch = torch.stack(padded_tensors)
                    else:
                        # All tensors have the same shape, can stack directly
                        batch = torch.stack(batch_tensors)
                
                # Move tensors to device
                batch = batch.to(self.device)
                if sbert_embeddings is not None:
                    sbert_embeddings = sbert_embeddings.to(self.device)
                
                # Forward pass with mixed precision
                optimizer.zero_grad()
                
                if scaler is not None:
                    # Use mixed precision training
                    with torch.cuda.amp.autocast():
                        logits, mean, logvar, vq_loss, _ = self(batch, sbert_embeddings, training=True)
                        loss, recon_loss, kl_loss, vq_loss_val = self.loss_function(batch, logits, mean, logvar, vq_loss)
                    
                    # Backward pass with scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    
                    # Update parameters with scaling
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training without mixed precision
                    logits, mean, logvar, vq_loss, _ = self(batch, sbert_embeddings, training=True)
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
                
                # Periodically print sample predictions (only once every 100 steps)
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
                    
                    # Save best model
                    best_model_path = os.path.join("checkpoints", "best_model.pt")
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, best_model_path)
                    print(f"New best model saved with validation loss: {best_loss:.4f}")
                else:
                    patience += 1
                    if patience >= max_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            # Validate on sample text periodically
            if (epoch + 1) % validate_every == 0 or epoch == epochs - 1:
                self.validate_on_samples()
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_every == 0:
                checkpoint_path = f"checkpoints/codec_epoch_{epoch+1}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        # Load the best model if it exists
        best_model_path = os.path.join("checkpoints", "best_model.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        
        return self
    
    def validate(self, val_dataloader):
        """Validate model on validation data"""
        self.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_data in val_dataloader:
                # Unpack data - can be either just tokens or (tokens, sbert_embeddings)
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    batch, sbert_embeddings = batch_data
                    sbert_embeddings = sbert_embeddings.to(self.device)
                else:
                    batch = batch_data
                    sbert_embeddings = None
                
                batch = batch.to(self.device)
                
                # Forward pass
                logits, mean, logvar, vq_loss, _ = self(batch, sbert_embeddings, training=False)
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
            "codebook_size": self.codebook_size,
            "sbert_model_name": self.sentence_encoder.get_config_dict()['model_name']
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
            device=device,
            sbert_model_name=config.get("sbert_model_name", "all-MiniLM-L6-v2")
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


# Custom dataset class that includes SBERT embeddings
class SBERTEnhancedDataset(Dataset):
    """Dataset that includes SBERT embeddings"""
    def __init__(self, sentences, embeddings, tokenizer, sequence_length=64):
        self.sentences = sentences
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        text = self.sentences[idx]
        tokens = self.tokenizer.encode(text, self.sequence_length)
        embedding = self.embeddings[idx]
        # Ensure tokens is a tensor
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        # Ensure embedding is a tensor
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float)
        return tokens, embedding

# Modify the training function to use SBERT
def train_enhanced_neural_codec(corpus_path=None, output_path="enhanced_neural_codec", 
                              epochs=30, continue_training=False, corpus_size=5000000,
                              sbert_model_name="all-MiniLM-L6-v2"):
    """Train an enhanced neural codec on a corpus with SBERT integration"""
    # Check for existing models
    if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "tokenizer")):
        print(f"Loading existing neural codec from {output_path}")
        
        # Load tokenizer
        tokenizer = BPETokenizer.load(os.path.join(output_path, "tokenizer"))
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=== Device Information ===")
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"Memory Usage: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
        print("=======================\n")
        
        # Load model
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
                
                checkpoint = torch.load(checkpoint_path, map_location=device)
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
        # Create a simplified corpus for demo purposes
        corpus_lines = []
        for i in range(1000):  # Reduce size for quicker testing
            corpus_lines.append(f"This is a sample text line {i} for training the neural codec model.")
        corpus = "\n".join(corpus_lines)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Device Information ===")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print("=======================\n")
    
    # Train BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=8000)
    tokenizer.train(corpus)
    
    # Load SBERT model
    print(f"Loading Sentence-BERT model: {sbert_model_name}")
    sentence_encoder = SentenceTransformer(sbert_model_name)
    sentence_encoder = sentence_encoder.to(device)
    
    # Split data into train and validation
    lines = [line for line in corpus.split("\n") if len(line.strip()) > 0]
    split_idx = int(len(lines) * 0.95)
    train_corpus = "\n".join(lines[:split_idx])
    val_corpus = "\n".join(lines[split_idx:])
    
    # Precompute SBERT embeddings for efficiency
    print("Precomputing SBERT embeddings for training data...")
    train_sentences = [line for line in train_corpus.split("\n") if len(line.strip()) > 0]
    
    # Process in smaller batches to reduce memory usage
    batch_size = 32
    train_embeddings = []
    
    for i in range(0, len(train_sentences), batch_size):
        batch = train_sentences[i:i+batch_size]
        with torch.no_grad():
            batch_embeddings = sentence_encoder.encode(batch, convert_to_tensor=True)
            train_embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    train_embeddings = torch.cat(train_embeddings, dim=0)
    
    # Do the same for validation data
    val_sentences = [line for line in val_corpus.split("\n") if len(line.strip()) > 0]
    val_embeddings = []
    
    for i in range(0, len(val_sentences), batch_size):
        batch = val_sentences[i:i+batch_size]
        with torch.no_grad():
            batch_embeddings = sentence_encoder.encode(batch, convert_to_tensor=True)
            val_embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    if val_embeddings:
        val_embeddings = torch.cat(val_embeddings, dim=0)
    
    # Create custom dataset class that includes SBERT embeddings
    class SBERTEnhancedDataset(Dataset):
        def __init__(self, sentences, embeddings, tokenizer, sequence_length=64):
            self.sentences = sentences
            self.embeddings = embeddings
            self.tokenizer = tokenizer
            self.sequence_length = sequence_length
        
        def __len__(self):
            return len(self.sentences)
        
        def __getitem__(self, idx):
            text = self.sentences[idx]
            tokens = self.tokenizer.encode(text, self.sequence_length)
            embedding = self.embeddings[idx]
            return tokens, embedding
    
    # Create datasets
    train_dataset = SBERTEnhancedDataset(train_sentences, train_embeddings, tokenizer, sequence_length=64)
    val_dataset = SBERTEnhancedDataset(val_sentences, val_embeddings, tokenizer, sequence_length=64) if len(val_sentences) > 0 else None
    
    # Create model with SBERT integration
    model = EnhancedNeuralTextCodec(
        vocab_size=len(tokenizer.tokenizer),
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=64,
        sequence_length=64,
        quantize=True,
        codebook_size=16,
        device=device,
        sbert_model_name=sbert_model_name
    )
    
    model.set_tokenizer(tokenizer)
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Vocabulary size: {len(tokenizer.tokenizer)}")
    print(f"Embedding dimension: {128}")
    print(f"Hidden dimension: {256}")
    print(f"Latent dimension: {64}")
    print(f"Sequence length: {64}")
    print(f"Codebook size: {16}")
    print(f"SBERT model: {sbert_model_name}")
    
    # Calculate model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 * 1024)
    print(f"Trainable model size: {model_size:.2f} MB")
    
    # Train the model
    model.train_model(
        train_dataset, 
        val_dataset=val_dataset,
        batch_size=32,  # Smaller batch size to reduce memory usage
        epochs=epochs,
        learning_rate=3e-4,  # Slightly reduced learning rate
        validate_every=1,
        checkpoint_every=1
    )
    
    # Save model and tokenizer
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    tokenizer.save(os.path.join(output_path, "tokenizer"))
    
    print(f"Saved enhanced neural codec to {output_path}")
    
    return model, 0

# Example usage
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Neural Text Codec with SBERT')
    parser.add_argument('--retrain', action='store_true', help='Force retraining even if model exists')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--corpus_size', type=int, default=1000000, help='Size of corpus in characters')
    parser.add_argument('--test_only', action='store_true', help='Skip training and only test the model')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from the last checkpoint')
    parser.add_argument('--sbert_model', type=str, default='all-MiniLM-L6-v2', 
                        help='SBERT model to use (default: all-MiniLM-L6-v2)')
    args = parser.parse_args()
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set model path
    model_path = "sbert_enhanced_neural_codec"
    
    # Check if we should retrain from scratch
    if args.retrain and os.path.exists(model_path):
        print("Removing existing model for retraining...")
        import shutil
        shutil.rmtree(model_path)
    
    # Train or load neural codec
    if not args.test_only:
        # Create corpus if needed
        corpus_path = "corpus.txt"
        if not os.path.exists(corpus_path) or args.retrain:
            print("Creating sample corpus...")
            with open(corpus_path, 'w', encoding='utf-8') as f:
                for i in range(10000):
                    f.write(f"This is sample sentence {i} for testing the neural codec model.\n")
        
        # Train the model
        model, start_epoch = train_enhanced_neural_codec(
            corpus_path, model_path, args.epochs, args.continue_training, args.corpus_size,
            sbert_model_name=args.sbert_model
        )
    else:
        # Load existing model
        if not os.path.exists(model_path):
            print("No existing model found. Please train a model first or remove --test_only flag.")
            exit(1)
        
        # Load tokenizer
        tokenizer = BPETokenizer.load(os.path.join(model_path, "tokenizer"))
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = EnhancedNeuralTextCodec.load(model_path, device)
        model.set_tokenizer(tokenizer)
    
    # Example messages for testing
    test_messages = [
        "The quick brown fox jumps over the lazy dog.",
        "Neural network encoders can achieve better compression ratios than traditional methods by learning language patterns.",
        "This is a test of the enhanced neural codec system with SBERT integration for better semantic understanding."
    ]
    
    # Test with each message
    for i, message in enumerate(test_messages):
        print(f"\n===== TESTING WITH MESSAGE {i+1} =====")
        print(f"Message: '{message}'")
        
        # Encode and decode
        encoded = model.encode(message)
        decoded = model.decode(encoded)
        
        # Print statistics
        stats = model.compression_stats(message)
        print(f"Original: {stats['original_size']} bytes")
        print(f"Compressed: {stats['compressed_size']} bytes")
        print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"Space saving: {stats['space_saving']:.2f}%")
        print(f"Decoded: '{decoded}'")
        
        # Calculate similarity
        original_words = set(message.lower().split())
        decoded_words = set(decoded.lower().split())
        
        if len(original_words) > 0:
            overlap = len(original_words.intersection(decoded_words))
            similarity = overlap / len(original_words)
            print(f"Word similarity: {similarity:.2f}")