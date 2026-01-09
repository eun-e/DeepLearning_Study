import torch
import torch.nn as nn

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(16)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.dropout1 = nn.Dropout2d(0.25)
    
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(32)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.dropout2 = nn.Dropout2d(0.25)

    self.fc1 = nn.Linear(32*8*8, 128)
    self.dropout3 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(128, 10)


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x) 
    x = self.pool1(x) 
    x = self.dropout1(x)

    x = self.conv2(x) 
    x = self.bn2(x) 
    x = self.relu2(x) 
    x = self.pool2(x) 
    x = self.dropout2(x) 

    # 평탄화
    x = x.view( x.size(0), -1 ) 
    # 완전 연결 레이어 
    x = self.fc1(x) 
    x = self.dropout3(x)
    x = self.fc2(x)
    return x
  
model = CNN()
print(model)
