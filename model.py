
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, batch_first=True)

    def forward(self, x):
        outputs, (h,c) = self.rnn(x)
        return outputs, (h,c)

class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden*2, 1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attn_weights = torch.softmax(energy.squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        return context

class Decoder(nn.Module):
    def __init__(self, hidden, output_dim):
        super().__init__()
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, output_dim)

    def forward(self, context, h, c):
        out, (h,c) = self.rnn(context, (h,c))
        pred = self.fc(out[:,-1])
        return pred, (h,c)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.att=attention

    def forward(self, x):
        enc_out,(h,c)=self.encoder(x)
        context=self.att(h, enc_out)
        y,(h,c)=self.decoder(context,h,c)
        return y
