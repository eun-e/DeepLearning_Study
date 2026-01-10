import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegressionModel(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(LinearRegressionModel, self).__init__()  # super는 부모 class의 기능을 가져와 초기화한다는 뜻
    self.linear = nn.Linear(input_dim, output_dim)
  
  def forward(self, x):
    return self.linear(x)
  
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_true = 2*x + 1

model = LinearRegressionModel(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
  optimizer.zero_grad()
  pred = model(x)
  loss = criterion(pred, y_true)
  loss.backward()
  optimizer.step()  # 파라미터 업데이트: 기울기와 학습률을 이용해 가중치를 조정함

  if (epoch+1)%20==0:
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
