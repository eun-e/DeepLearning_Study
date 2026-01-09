# Sobel Filter로 이미지 내에서 색이 갑자기 변하는 경계선을 찾아내는 역할을 함

filter_2d = torch.tensor([[
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0]
]]).unsqueeze(0)

sample_image_2 = torch.zeros(1,1,8,8)
sample_image_2[0.0.:,3:5] = 1.0   # 3~4열만 하얀색(1.0)으로 칠한 세로 기둥 모양
output_2 = nn.functional.conv2d(sample_image_2, filter_2d)


# Stride와 Padding
import torch.nn.functional as F

input_image = torch.randn(1,1,6,6)    # 1개의 배치, 1개의 채널, 6X6 크기
filter = torch.randn(1,1,3,3)
output = F.conv2d(input_image, filter, stride=1, padding=1)

