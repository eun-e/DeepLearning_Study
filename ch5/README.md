## Chapter 5-1. 합성곱 신경망의 기본 개념과 구조

#### 🔍 개념 정리
- DNN: 입력층 - 여러 은닉층 - 출력층 (Fully Connected Layer만 사용) <br>
  └ 입력 데이터를 1차원 벡터로 펼쳐서(flatten) 사용함
- CNN: Convolution layer, Pooling layer 거치고 마지막에 Fully Connected layer 사용 <br>
  └ 필터로 일부 영역만 보고 같은 필터를 전체에 적용해 가중치를 공유함 <br>
  └ 입력 이미지에서 객체 위치가 달라져도 동일한 특징을 감지할 수 있음
<br>


## Chapter 5-2. 합성곱 신경망의 주요 구성 요소

#### 🔍 개념 정리
- Convolution layer: 입력 데이터에 필터를 적용하여 특징을 추출
- CNN의 수직 에지 필터: [[1,0,-1],[1,0,-1],[1,0,-1]] 계산하면 결국 (왼쪽 값-오른쪽 값) <br>
  └ 보통 CNN에서 어두운 픽셀은 값이 작고(0) 밝은 픽셀은 값이 큼(1)
- Stride: 필터가 이동하는 간격을 결정함 <br>
  └ 출력 크기 = ( 입력크기 - 필터크기 + 2*패딩 )/스트라이드 + 1
- 필터의 shape = (출력 채널, 입력 채널(depth), feature_size)
- Pooling layer: 특성 맵의 공간적 크기를 줄이는 다운 샘플링 연산을 수행함 <br>
  └ max pooling, average pooling
- 배치 정규화는 일반적으로 합성곱 레이어와 활성화 함수 사이에 위치함


#### ❓헷갈리는 내용 정리
- p156; 어두운 영역에서 밝은 영역으로 변할 때 양수 값이 생성된다. <br>
  └ 필터의 오른쪽이 흰색을 만날 때 음수 값이 나왔다가 필터의 왼쪽이 흰색을 만나면 양수가 됨 <br>
    즉!! 왼쪽 가중치를 중심으로 생각했을 때 어두운 영역에서 밝은 영역이 되면 양수 값이 됨
- F.relu(x) vs nn.ReLU() <br>
  └ F는 함수 호출 방식으로 nn.Sequential 사용 불가능, 모델 구조에 안 보임
- 풀링 크기와 스트라이드를 동일하게 설정하여 겹치지 않게 함 - 왜? 다운샘플링 목적 이루기 위해
  

#### 📝 practice2.py
````text
CNN(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout1): Dropout2d(p=0.25, inplace=False)
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout2): Dropout2d(p=0.25, inplace=False)
  (fc1): Linear(in_features=2048, out_features=128, bias=True)
  (dropout3): Dropout(p=0.5, inplace=False)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
````
<br>

## Chapter 5-3. 고급 합성곱 신경망 아키텍처

#### 🔍 개념 정리
- VCG 네트워크: 3X3 합성곱 필터를 반복해서 사용하며, 풀링을 통해 특성 맵의 크기를 점진적으로 줄여나감 <br>
  └ 합성곱 층의 stride는 1로 설정함
- Conv2d의 파라미터 수: 가중치는 Cout X Cin X k X k, bias는 Cout

<br>

## Chapter 5-4. ResNet 구조

#### 🔍 개념 정리
- 깊은 네트워크를 효과적으로 학습시키기 위해 잔차 학습 개념을 도입함
- 직접 연결: 입력을 직접 출력에 더하는 연결 → 기울기 소실 문제 완화, 최적화 용이성, 성능 향상
