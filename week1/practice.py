import torch

def main():
  x = torch.tensor([1.0, 2.0, 3.0])
  print('Hello, Pytorch!')
  print(f'Tensor x: {x}')

main()

# Hello, Pytorch!
# Tensor x: tensor([1., 2., 3.])

=============================================================================

x= torch.tensor(2.0, requires_grad=True)   # 자동 미분 활성화
y = x**2 + 3*x + 1
y.backward()                               # 자동으로 미분 계산
