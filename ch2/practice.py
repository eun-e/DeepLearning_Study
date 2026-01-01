# 특수 함수로 텐서 생성

zeros_tensor = torch.zeros((3,3))
ones_tensor = torch.ones((2,4))
full_tensor = torch.full((2,2), 7) 

arange_tensor = torch.arange(0,10, step=2)  
linspace_tensor = torch.linspace(0,1, steps=5)  # 중요 포인트!! step은 간격, steps는 개수 의미

# tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
# tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]])
# tensor([[7, 7],
        [7, 7]])
# tensor([0, 2, 4, 6, 8])
# tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

