class TextEncoder(nn.Module):
    def __init__(self, d_embedding, vocab_size, max_len):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embedding) # Assuming padding_idx=0 for BERT tokenizer
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, d_embedding))
        # TransformerEncoderLayer and TransformerEncoder default to batch_first=False
        # Input: (S, N, E) = (seq_len, batch_size, d_embedding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_embedding, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x_ids): # x_ids shape: (batch_size, seq_len)
        x_embedded = self.embedding(x_ids)  # Shape: (batch_size, seq_len, d_embedding)
        
        # Add positional encoding (sliced to current seq_len)
        # Ensure x_ids.shape[1] does not exceed self.positional_encoding.shape[1] (max_len)
        # current_seq_len = x_ids.shape[1]
        x = x_embedded + self.positional_encoding # Shape: (batch_size, seq_len, d_embedding)
        
        x = x.transpose(0, 1)  # Shape: (seq_len, batch_size, d_embedding)
        x = self.encoder(x)     # Shape: (seq_len, batch_size, d_embedding)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, d_embedding, resnet101, text_encoder, num_classes):
        super(EncoderDecoder, self).__init__()
        if d_embedding % 8 != 0: # num_heads for MultiheadAttention is 8
            raise ValueError(f"d_embedding ({d_embedding}) must be divisible by num_heads (8).")
            
        self.encoder = resnet101
        self.projection = nn.Linear(2048, d_embedding)
        self.text_encoder = text_encoder
        
        # MultiheadAttention defaults to batch_first=False
        # Query: (L, N, E), Key: (S, N, E), Value: (S, N, E)
        # L=target_seq_len (text), S=source_seq_len (image features)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_embedding, num_heads=8, batch_first=False)
        
        self.layer_1 = nn.Linear(d_embedding, 500)
        self.layer_2 = nn.Linear(500, num_classes)
        self.relu = nn.ReLU()

    def forward(self, image, text):
        # Image processing
        # x shape: (N, 2048, H_feat, W_feat) e.g. (N, 2048, 7, 7) for 224x224 input to ResNet
        x = self.encoder(image)
        x = x.permute(0, 2, 3, 1)  # Shape: (N, H_feat, W_feat, 2048)
        # Flatten spatial dimensions: (N, H_feat*W_feat, 2048)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], 2048)
        x = self.projection(x).permute(1, 0, 2)
        # Transpose for attention: (H_feat*W_feat, N, d_embedding)
        # img_key_value = img_features.

        # Text processing
        # text_features shape: (S_text, N, d_embedding)
        y = self.text_encoder(text)

        # Cross-attention
        # query=text_features (S_text, N, d_embedding)
        # key=img_key_value (S_img, N, d_embedding)
        # value=img_key_value (S_img, N, d_embedding)
        # attn_output shape: (S_text, N, d_embedding)
        attn_output, _ = self.cross_attention(query=y, key=x, value=x)
        
        # Use the representation of the first token (e.g., CLS token) from text sequence after attention
        # cls_representation shape: (N, d_embedding)
        attn_output = attn_output[0]#cls token
        out = self.layer_1(attn_output)
        out = self.relu(out)
        out = self.layer_2(out)
        return out
