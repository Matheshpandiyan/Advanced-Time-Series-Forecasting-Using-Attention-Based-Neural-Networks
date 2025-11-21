import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h, c) = self.lstm(x)
        last = out[:, -1, :]        # (batch, hidden_size)
        return self.linear(last)    # (batch, output_size)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=128, dropout=0.1, output_size=1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, src, tgt=None):
        # src: (batch, src_seq, input_size)
        bs = src.size(0)
        src = self.input_proj(src) * (self.d_model ** 0.5)
        src = self.pos_enc(src)
        enc = self.encoder(src)  # (batch, src_seq, d_model)

        if tgt is None:
            # autoregressive single-step: use last encoder vector as query
            last = enc[:, -1:, :]  # (batch, 1, d_model)
            # decoder expects (tgt, memory) shapes handled by batch_first=True
            dec = self.decoder(last, enc)
            out = self.out(dec[:, -1, :])
            return out
        else:
            tgt = self.input_proj(tgt) * (self.d_model ** 0.5)
            tgt = self.pos_enc(tgt)
            dec = self.decoder(tgt, enc)
            return self.out(dec[:, -1, :])
