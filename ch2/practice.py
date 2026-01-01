# 특수 함수로 텐서 생성

zeros_tensor = torch.zeros((3,3))
ones_tensor = torch.ones((2,4))
full_tensor = torch.full((2,2), 7)  # 2X2 텐서를 7로 전부 채움

arrange_tensor = torch.arrange(0,10, step=2)   # 0, 2, 4, 6, 8
linspace_tensor = torch.linspace(0,1, step=5)  # 0, 0.25, 0.5, 0.75, 1
