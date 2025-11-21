
import torch
import torch.nn as nn
import pandas as pd
from model import Encoder, Decoder, Attention, Seq2Seq

df = pd.read_csv('series.csv')
data = torch.tensor(df.values, dtype=torch.float32)
seq_len=20
X=[]
Y=[]
for i in range(len(data)-seq_len):
    X.append(data[i:i+seq_len,:2])
    Y.append(data[i+seq_len,2])
X=torch.stack(X)
Y=torch.tensor(Y).float()

encoder=Encoder(2,64)
att=Attention(64)
decoder=Decoder(64,1)
model=Seq2Seq(encoder,decoder,att)
opt=torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn=nn.MSELoss()

for epoch in range(5):
    model.train()
    opt.zero_grad()
    pred=model(X)
    loss=loss_fn(pred.squeeze(),Y)
    loss.backward()
    opt.step()
    print(epoch, loss.item())
