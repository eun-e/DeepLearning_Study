import torch 
import torch.nn as nn 
import torch.optim as optim 

# 간단한 인코더 정의 
class Encoder(nn.Module): 
  def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=1): 
    super(Encoder, self).__init__() 
    self.embedding = nn.Embedding(input_dim, emb_dim) 
    self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True) 
  def forward(self, src): 
    # src: [batch_size, src_len] -> batch_size는 한 번에 처리할 수 있는 문장의 개수, src_len은 문장의 길이
    embedded = self.embedding(src) # [batch_size, src_len, emb_dim] -> 각 단어 번호를 고차원의 벡터로 변
    outputs, (hidden, cell) = self.lstm(embedded) 
    return outputs, (hidden, cell) 
    
# 간단한 디코더 정의 
class Decoder(nn.Module) : 
  def __init__(self, output_dim, emb_dim, hidden_dim, n_layers=1): 
  super(Decoder, self).__init__() 
    self.embedding = nn.Embedding(output_dim, emb_dim) 
    self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, batch_first=True) 
    self.fc_out = nn.Linear(hidden_dim, output_dim) 
  def forward(self, trg, hidden, cell): 
    # trg: [batch_size](한 시점의 단어 인텍스) 
    trg = trg.unsqueeze(1) # [batch_size, 1] 
    embedded = self.embedding(trg) # [batch_size, 1, emb_dim] 
    output, (hidden , cell) = self.lstm(embedded, (hidden , cell)) 
    prediction = self.fc_out(output.squeeze(1)) # [batch_size, output_dim] 
    return prediction, hidden, cell 

#간단한 seq2seq 모델 정의
class Seq2seq(nn.Module):
  def __init__(self, encoder, decoder, device): 
    super(Seq2Seq , self).__init__() 
    self.encoder = encoder 
    self.decoder = decoder 
    self.device = device 

  def forward(self, src, trg, teacher_forcing_ratio=0.5): 
    batch_size = src.size(0) 
    trg_len = trg.size(1) 
    output_dim = self.decoder.fc_out.out_features 
    outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device) 
    enc_outputs, (hidden, cell) = self.encoder(src) 
    input = trg[:, 0] # 시작 토큰, 보통 <sos>를 넣어줌 
    for t in range(1, trg_len): 
      output, hidden, cell = self.decoder(input, hidden, cell) 
      outputs[:, t, :] = output 
      teacher_force = torch.rand(1).item() < teacher_forcing_ratio 
      top1 = output.argmax(1) 
      input = trg[:, t] if teacher_force else top1 
    return outputs 
  
