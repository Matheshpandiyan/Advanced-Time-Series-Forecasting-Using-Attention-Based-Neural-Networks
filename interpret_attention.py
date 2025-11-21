import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def plot_attention_matrix(attn_weights, save_path=None, title='Attention'):
    """attn_weights: numpy array (tgt_len, src_len) or (num_heads, tgt_len, src_len)"""
    arr = attn_weights
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    plt.figure(figsize=(6,4))
    plt.imshow(arr, aspect='auto')
    plt.xlabel('Source time')
    plt.ylabel('Target time')
    plt.title(title)
    plt.colorbar()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def extract_attention_example(model, src_tensor):
    """Run a forward pass and try to extract attention weights by re-computing attention
    from the encoder outputs using a simple MultiheadAttention to inspect weights.
    Returns attn_weights (num_heads, tgt_len, src_len) averaged across heads may be used."""
    model.eval()
    with torch.no_grad():
        # get encoder representations
        src = src_tensor.unsqueeze(0)  # (1, seq, feat)
        enc_in = model.input_proj(src) * (model.d_model ** 0.5)
        enc_in = model.pos_enc(enc_in)
        enc = model.encoder(enc_in)  # (1, seq, d_model)
        # Use a single MultiheadAttention to compute attention of last step over encoder memory
        mha = torch.nn.MultiheadAttention(embed_dim=model.d_model, num_heads=4, batch_first=True)
        query = enc[:, -1:, :]  # (1, 1, d_model)
        attn_out, attn_w = mha(query, enc, enc, need_weights=True)
        # attn_w shape: (batch, tgt_len, src_len)
        return attn_w.squeeze(0).cpu().numpy()
