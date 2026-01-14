hidden_size = 20
num_layers = 1
lstm  = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

x = torch.randn(32, 10, input_size)
output, (h_n, c_n) = lstm(x)

