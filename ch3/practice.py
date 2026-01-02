# ---------- nn.Module 클래스를 이용한 신경망 구현------------

# 데이터셋 생성
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 신경망 모델 정의
class SimpleNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(SimpleNN, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, output_size)
  
  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x

# 모델 초기화
model = SimpleNN(input_size=10, hidden_size=20, output_size=10)
print(model)

# 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.paramters, lr=0.001)

# 학습 루프 구현
for epoch in range(10):
  for batch in dataloader:
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
  print(f'Epoch {epoch+1}, Los: {loss.item()}')

# 모델 저장 및 불러오기
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# 학습 중 NaN값 확인 및 학습률 조정
if torch.isnan(loss).any():
  print('Loss contains NaN values! Reducing learning rate...')
  for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1

# 데이터 증강 기법 적용(이미지 데이터의 경우)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
