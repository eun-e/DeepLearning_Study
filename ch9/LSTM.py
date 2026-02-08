import torch.nn as nn 
class SentimentLSTM(nn .Module): 
  def __init__(self, vocab_size, embedding_dim , hidden_dim, output_dim, n_layers=l , dropout=0.5): 
    super(SentimentLSTM, self).__init__() 
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) 
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True , dropout=dropout) 
    self.fc = nn.Linear(hidden_dim, output_dim) 
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x): 
    # x: [batch_size, seq_length] 
    embedded = self.dropout(self.embedding(x)) # [batch_size, seq_length, embedding_dim] 
    lstm_out, (hidden, cell) = self.lstm(embedded) 
    final_feature = hidden [-1]  # 마지막 레이어의 마지막 hidden state, shape: [batch_size, hidden_dim ] 
    output = self.fc(final_feature) # [batch_size, output_dim] 
    return output

vocab_size = 5000
embedding_dim = 100
hidden_dim = 128
output_dim = 2

model_lstm = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_lstm.parameteres(), lr=0.001)
print(model_lstm)
