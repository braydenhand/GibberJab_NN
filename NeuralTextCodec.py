import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import os
import pickle
import time
import tempfile
import subprocess

class SubwordTokenizer:
    """Simple subword tokenizer similar to BPE (Byte-Pair Encoding)"""
    
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.word_vocab = {}  # Word to ID mapping
        self.id_vocab = {}    # ID to word mapping
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3
        }
        
    def train(self, corpus):
        """Train tokenizer on a text corpus"""
        print("Training tokenizer...")
        
        # Start with special tokens
        self.word_vocab = self.special_tokens.copy()
        self.id_vocab = {v: k for k, v in self.word_vocab.items()}
        
        # Simple word-level tokenization for demo purposes
        # In production, you'd implement proper BPE (Byte-Pair Encoding)
        words = re.findall(r'\b\w+\b', corpus.lower())
        word_counts = Counter(words)
        
        # Add most common words to vocabulary
        idx = len(self.word_vocab)
        for word, _ in word_counts.most_common(self.vocab_size - len(self.special_tokens)):
            if word not in self.word_vocab:
                self.word_vocab[word] = idx
                self.id_vocab[idx] = word
                idx += 1
                
                if idx >= self.vocab_size:
                    break
        
        print(f"Tokenizer vocabulary size: {len(self.word_vocab)}")
        return self
    
    def encode(self, text, max_length=None):
        """Convert text to token IDs"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Convert words to token IDs
        token_ids = [self.special_tokens["[BOS]"]]
        for word in words:
            if word in self.word_vocab:
                token_ids.append(self.word_vocab[word])
            else:
                token_ids.append(self.special_tokens["[UNK]"])
        token_ids.append(self.special_tokens["[EOS]"])
        
        # Pad or truncate to max_length if specified
        if max_length is not None:
            if len(token_ids) < max_length:
                token_ids = token_ids + [self.special_tokens["[PAD]"]] * (max_length - len(token_ids))
            else:
                token_ids = token_ids[:max_length-1] + [self.special_tokens["[EOS]"]]
        
        # Return tensor with shape [sequence_length]
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        words = []
        for token_id in token_ids:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
                
            # Skip special tokens
            if token_id in [self.special_tokens["[PAD]"], self.special_tokens["[BOS]"], 
                          self.special_tokens["[EOS]"]]:
                continue
            
            if token_id in self.id_vocab:
                words.append(self.id_vocab[token_id])
            else:
                words.append("[UNK]")
        
        return " ".join(words)
    
    def save(self, path):
        """Save tokenizer to file"""
        data = {
            "word_vocab": self.word_vocab,
            "id_vocab": self.id_vocab,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from file"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.word_vocab = data["word_vocab"]
        tokenizer.id_vocab = data["id_vocab"]
        tokenizer.special_tokens = data["special_tokens"]
        
        return tokenizer


class TextDataset(Dataset):
    """Dataset for training the neural codec"""
    
    def __init__(self, corpus, tokenizer, sequence_length=64):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        
        # Split corpus into sentences
        self.sentences = [s.strip() for s in corpus.split(".") if len(s.strip()) > 10]
        print(f"Dataset created with {len(self.sentences)} sentences")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        text = self.sentences[idx]
        tokens = self.tokenizer.encode(text, self.sequence_length)
        # Ensure the tensor has the correct shape [sequence_length]
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)  # Add batch dimension
        return tokens.squeeze(0)  # Remove batch dimension to get [sequence_length]


class Encoder(nn.Module):
    """VAE Encoder module"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, x):
        # Embed tokens
        embedded = self.embedding(x)
        
        # Pass through LSTM
        output, (hidden, _) = self.lstm(embedded)
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Get mean and log variance for latent space
        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)
        
        # Sample from latent space
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z, mean, logvar


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for latent space"""
    
    def __init__(self, latent_dim, n_embeddings=16, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings
        self.commitment_cost = commitment_cost
        
        # Create codebook embeddings - one codebook per dimension
        # This is the key difference from the original implementation
        # Each dimension has its own separate codebook
        self.codebooks = nn.ModuleList([
            nn.Embedding(n_embeddings, 1) for _ in range(latent_dim)
        ])
        
        # Initialize codebook values
        for codebook in self.codebooks:
            codebook.weight.data.uniform_(-1./n_embeddings, 1./n_embeddings)
    
    def forward(self, z):
        # z has shape [batch_size, latent_dim]
        batch_size = z.shape[0]
        
        # Quantize each dimension separately
        z_q = torch.zeros_like(z)
        indices = torch.zeros(batch_size, self.latent_dim, dtype=torch.long, device=z.device)
        vq_loss = 0.0
        
        for i in range(self.latent_dim):
            # Get the values for this dimension
            z_dim = z[:, i].unsqueeze(1)  # [batch_size, 1]
            
            # Calculate distances to codebook entries
            codebook = self.codebooks[i]
            # Expand codebook weights to match batch dimension
            codebook_weights = codebook.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_embeddings, 1]
            z_dim_expanded = z_dim.unsqueeze(1).expand(-1, self.n_embeddings, -1)  # [batch_size, n_embeddings, 1]
            
            # Calculate distances
            distances = torch.sum((z_dim_expanded - codebook_weights) ** 2, dim=2)  # [batch_size, n_embeddings]
            
            # Get closest codebook entry
            min_indices = torch.argmin(distances, dim=1)  # [batch_size]
            indices[:, i] = min_indices
            
            # Get quantized values
            z_q_dim = codebook(min_indices)  # [batch_size, 1]
            z_q[:, i] = z_q_dim.squeeze(1)
            
            # Compute VQ loss for this dimension
            e_latent_loss = F.mse_loss(z_q_dim.detach(), z_dim)
            q_latent_loss = F.mse_loss(z_q_dim, z_dim.detach())
            vq_loss += q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Use straight-through estimator for entire vector
        z_q_sg = z + (z_q - z).detach()
        
        return z_q_sg, vq_loss, indices


class Decoder(nn.Module):
    """VAE Decoder module"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, sequence_length):
        super(Decoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, z):
        # Transform latent vector to hidden state
        hidden = self.latent_to_hidden(z)
        
        # Repeat to create sequence
        hidden = hidden.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Pass through LSTM
        output, _ = self.lstm(hidden)
        
        # Get token probabilities
        logits = self.output(output)
        
        return logits


class NeuralTextCodec(nn.Module):
    """Neural network-based text codec optimized for FSK transmission"""
    
    def __init__(self, vocab_size=8000, embedding_dim=128, hidden_dim=256, 
                latent_dim=128, sequence_length=64, quantize=True, 
                codebook_size=16, device="cpu"):
        super(NeuralTextCodec, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.codebook_size = codebook_size
        self.device = device
        
        # Calculate bits per dimension based on codebook size
        self.bits_per_dim = int(np.log2(codebook_size))
        print(f"Using {self.bits_per_dim} bits per dimension with latent_dim={latent_dim}")
        print(f"This allows for approximately {latent_dim * self.bits_per_dim / 8:.1f} bytes per message")
        
        # Encoder and decoder models
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, latent_dim, sequence_length)
        
        # Vector quantizer (if enabled)
        self.quantize = quantize
        if quantize:
            self.vector_quantizer = VectorQuantizer(latent_dim, n_embeddings=codebook_size)
        else:
            self.vector_quantizer = None
        
        # Tokenizer
        self.tokenizer = None
        
        # Move to device
        self.to(device)
    
    def forward(self, x):
        # Encode
        z, mean, logvar = self.encoder(x)
        
        # Quantize if enabled
        if self.quantize:
            z_q, vq_loss, encoding_indices = self.vector_quantizer(z)
        else:
            z_q = z
            vq_loss = 0
            encoding_indices = None
        
        # Decode
        logits = self.decoder(z_q)
        
        return logits, mean, logvar, vq_loss, encoding_indices
    
    def loss_function(self, x, logits, mean, logvar, vq_loss=0):
        # Reconstruction loss (cross entropy)
        recon_loss = F.cross_entropy(logits.view(-1, self.vocab_size), x.view(-1), ignore_index=0)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / (x.size(0) * x.size(1))
        
        # Total loss
        return recon_loss + 0.1 * kl_loss + vq_loss
    
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
                _, _, encoding_indices = self.vector_quantizer(z)
                
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
            indices_tensor = torch.tensor(indices, device=self.device).long()
            
            # Reconstruct latent vector using the vector quantizer's codebooks
            z_q = torch.zeros(1, self.latent_dim, device=self.device)
            for i in range(self.latent_dim):
                if i < len(indices):
                    idx = indices_tensor[i]
                    if idx < self.codebook_size:
                        z_q[0, i] = self.vector_quantizer.codebooks[i](idx.unsqueeze(0)).squeeze()
        else:
            # Without quantization, convert bytes back to floats
            z_np = np.frombuffer(byte_data, dtype=np.uint8).astype(np.float32)
            
            # Rescale from 0-255 to -3 to 3
            z_np = z_np / 42.5 - 3
            
            # Ensure z_np length is compatible with latent_dim
            if len(z_np) > self.latent_dim:
                z_np = z_np[:self.latent_dim]
            elif len(z_np) < self.latent_dim:
                z_np = np.pad(z_np, (0, self.latent_dim - len(z_np)))
            
            z_q = torch.tensor(z_np, device=self.device).unsqueeze(0)
        
        # Decode
        with torch.no_grad():
            logits = self.decoder(z_q)
            token_ids = torch.argmax(logits, dim=2)[0]
        
        # Convert token IDs back to text
        return self.tokenizer.decode(token_ids)
    
    def set_tokenizer(self, tokenizer):
        """Set the tokenizer to use for encoding/decoding"""
        self.tokenizer = tokenizer
        return self
    
    def train_model(self, train_dataset, batch_size=64, epochs=10, learning_rate=1e-3):
        """Train the model on a dataset"""
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        print(f"Training model for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            for batch in dataloader:
                # Ensure batch has shape [batch_size, sequence_length]
                if len(batch.shape) == 1:
                    batch = batch.unsqueeze(0)  # Add batch dimension if missing
                batch = batch.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits, mean, logvar, vq_loss, _ = self(batch)
                
                # Calculate loss
                loss = self.loss_function(batch, logits, mean, logvar, vq_loss)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Validate on a sample after each epoch
            if epoch % 5 == 0 or epoch == epochs - 1:
                self.validate_on_sample("The quick brown fox jumps over the lazy dog.")
        
        return self
    
    def validate_on_sample(self, sample_text):
        """Test compression/decompression on a sample text"""
        print(f"\nValidation on: '{sample_text}'")
        
        # Encode and decode
        encoded = self.encode(sample_text)
        decoded = self.decode(encoded)
        
        print(f"Original: '{sample_text}'")
        print(f"Decoded:  '{decoded}'")
        print(f"Bytes:    {len(encoded)}")
        print()
        
        return decoded
    
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


def train_neural_codec(corpus_path=None, output_path="pytorch_neural_codec"):
    """Train a neural codec on a corpus"""
    # Check for existing models
    if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, "tokenizer.pkl")):
        print(f"Loading existing neural codec from {output_path}")
        
        # Load tokenizer
        tokenizer = SubwordTokenizer.load(os.path.join(output_path, "tokenizer.pkl"))
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NeuralTextCodec.load(output_path, device)
        model.set_tokenizer(tokenizer)
        
        return model
    
    # Create or load corpus
    if corpus_path and os.path.exists(corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = f.read()
    else:
        _, corpus = create_deep_corpus()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train tokenizer
    tokenizer = SubwordTokenizer(vocab_size=8000)
    tokenizer.train(corpus)
    
    # Create dataset
    dataset = TextDataset(corpus, tokenizer)
    
    # Create model with larger latent dimension
    model = NeuralTextCodec(
        vocab_size=len(tokenizer.word_vocab),
        embedding_dim=128,
        hidden_dim=256,
        latent_dim=128,  # Increased from 32 to 128
        sequence_length=64,
        quantize=True,
        codebook_size=16,  # 4 bits per dimension
        device=device
    )
    
    model.set_tokenizer(tokenizer)
    
    # Train the model for more epochs
    model.train_model(dataset, batch_size=32, epochs=10)
    
    # Test on sample text
    print("\nTesting codec on sample text:")
    sample_text = "The quick brown fox jumps over the lazy dog."
    model.validate_on_sample(sample_text)
    
    # Save model and tokenizer
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    tokenizer.save(os.path.join(output_path, "tokenizer.pkl"))
    
    print(f"Saved neural codec to {output_path}")
    
    return model
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
    
    try:
        # Initialize ggwave
        instance = ggwave.init()
        
        # Encode with ggwave
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
# Example usage
if __name__ == "__main__":
    # Train or load neural codec
    model = train_neural_codec()
    
    # Example message
    example_message = """Neural network approaches to text compression can achieve much better efficiency than traditional methods by learning the statistical and semantic patterns in language."""
    
    # Send the message
    send_neural_encoded_message(example_message, model)