import torch

def forward_pass(x, W, b):
  z = torch.mm(x, W) + b
  output = torch.relu(z)
  return output

x = torch.tensor([[1.0, 2.0],   # 주의할 점!! 멋대로 1.0이 아니라 1로 쓰면 0.5랑 type이 달라져서 오류
                 [3.0, 4.0]])   # 주의할 점!! tensor는 [[]] 이렇게 이중 괄호로 묶여 있음
W = torch.tensor([[0.5, -0.5],
                 [1.0, 1.0]])
b = torch.tensor([[1.0, 1.0]])

output = forward_pass(x, W, b)
print(output))
