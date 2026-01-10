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
  └ 출력 크기 = ( 입력크기 - 필터크기 + 2*패딩 ) / 스트라이드 + 1
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

<br>

## Chapter 5-5. 합성곱 신경망의 성능 최적화

#### 🔍 개념 정리
- 학습률 스케줄링: lr을 상황에 따라 바꾸면서 학습 속도를 조정함 <br>
  └  StepLR(demo_optimizer, step_size=30, gamma=0.1): 30 epoch마다 학습률을 0.1배로 줄이기 <br>
  └  ReduceLRPlateau(demo_optimizer, mode='min', factor=0.1, patience=10) <br>
    → mode는 loss가 줄어드는지 감시, patience는 10 epoch동안 개선 없으면 factor=0.1배로 감소 <br>
  └  ConsineAnnealingLR(demo_optimizer, T_max=200): 200 epoch동안 한 사이
- 가중치 초기화: 학습 시작 전에 신경망의 가중치를 어떤 값으로 시작할지 정하는 것 <br>
  └  Kaiming(He) 초기화: ReLU 쓸 때 최적화됨 <br>
  └  Xavier 초기화: sigmoid나 tanh 활성화 함수와 쓸 때 최적화됨 <br>
  └  정규 분포 초기화: 평균과 표준 편차를 지정하여 초기화함
- 최적화 알고리즘 <br>
  └  SGD: 가장 기본적인 알고맂므으로, 모멘텀을 추가하면 학습이 더 안정적이게됨 <br>
  └  Adam: 적응적 학습률을 사용해 파라미터마다 다른 학습률을 적용함 <br>
  └  RMSprop: 이동 평균을 사용해 기울기를 정규화함, 순환 신경망에서 효과 
- 양자화: 모델의 가중치와 활성화 값을 정밀도가 낮은 데이터 타입으로 변환해 모델 크기를 줄임
- 지식 증류: 큰 모델의 지식을 작은 모델로 전달하는 기법


#### ❓헷갈리는 개념들
- nn.Conv2d(in_channels, out_channels, kernel_size) 형태는 항상 바뀌지 않는데 첫 번째 Conv일 때만 앞에 이미지 채널(RGB, 흑백)이 옴 → 어차피 다 input 개념임, 두 번째 Conv부터는 이전 Conv가 만든 feature map 크기

<br>

## Chapter 5-6. 합성곱 신경망의 시각화와 해석

#### 🔍 개념 정리
- 중간층의 활성화: 각 층은 입력 이미지의 서로 다른 특징(에지, 텍스처, 패턴 등)을 추출함
- CAM: 합성곱 신경망이 분류 결과를 도출할 때 이미지의 어떤 부분에 주목했는지를 시각화하는 기 
<img width="752" height="388" alt="image" src="https://github.com/user-attachments/assets/347cf8e0-4581-4cbf-b18f-272819b2539e" />
- 입력 데이터가 각 층을 통과하며 나타난 반응 결과
  1. 한 칸은 입력 이미지가 특정 필터를 통과한 후의 결과물
  2. 밝은 곳은 강하게 반응한 곳이고 어두운 곳은 거의 반응 안 한 부분
  3. covn1에서는 미세한 엣지(선) 정보, conv2에서는 부분적인 모양 정보, conv3에서는 합쳐진 물체의 모양, 질감 등 점점 커짐
<img width="798" height="759" alt="image" src="https://github.com/user-attachments/assets/4de9c7aa-1f71-4f13-9d8b-dae9a7391a5d" />
- 숫자 배열을 시각화한 결과
  1. 한 칸은 가중치 배열 하나를 그림으로 바꾼 것
  2. 입력 이미지와 겹쳐서 계산할 때 어떤 특징에 반응할지를 결정하는 검색 패턴이 됨 

#### ❓헷갈리는 내용 정리
- ***질문 사항*** : p221에서 코드를 보면 conv1에 대해서는 맵을 저장 안 한걸로 보이는데 결과보면 conv1에 대해서도 시각화가 되어있어서..... 뭔가 놓친 부분이 있는 것 같은데 그게 뭔지 모르겠네요....












  
