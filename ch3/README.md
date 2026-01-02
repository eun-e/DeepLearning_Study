## Chapter 3-1. 신경망의 개요

#### 🔍 인공뉴런
- 여러 입력값을 가중치와 함께 조합하여 연산을 수행하고, 특정 활성화 함수를 적용해 최종 출력을 생산함
- 가중치 매개 변수와 편향값을 포함하며, 학습 과정을 통해 조정됨 <br>
  → 가중치는 입력값의 중요도를 결정하며, 학습 과정에서 최적의 값을 찾음 <br>
  → 편향은 활성화 함수를 거치기 전에 일정한 조정을 수행하여 뉴런의 활성화 여부를 더 정밀하게 조정함
- 활성화 함수: ReLU, Sigmoid, Tanh 등이 있으며 선형 연산을 비선형 변환함

#### 🔍 신경망의 종류
1. MLP: 여러 개의 은닉층으로 구성된 완전 연결 신경망 → 공간적, 시간적 구조를 학습하는 능력이 부족해 이미지나 시계열 데이터 처리에는 적합X
2. CNN: 이미지 데이터와 같은 2차원의 구조화된 데이터에 적합도록 설계된 신경망 → convolution과 pooling 연산을 통해 특징을 추출함
3. RNN: 시계열 데이터를 다루는 데 특화된 신경망 → 이전 상태의 정보를 저장해 시간에 따라 변하는 데이터를 효과적으로 처리함

#### 🔍 신경망의 응용
1. CV: 이미지 및 영상 데이터를 처리해 의미 있는 정보를 추출하는 분야 - CNN 사용
2. NLP: 인간의 언어를 기계가 이해하고 처리하는 기술 - RNN, LSTM, Transformer 사용

<br>

## Chapter 3-2. 기본신경망 구조의 이해

#### 🔍 단일 퍼셉트론의 기본 모델
<img width="250" height="110" alt="image" src="https://github.com/user-attachments/assets/16dc5dbc-7a2e-4f26-8ae3-07ba0f6bb599" /> <br>
- 가중치(w): 초기에는 랜덤 값으로 설정되며, 경사 하강법과 같은 알고리즘을 통해 학습이 진행될수록 업데이트됨 → 과적합인 경우 가중치 정규화
- 편향: 뉴런이 일정한 출력을 생성할 수 있도록 도움 if 편향이 0인 경우, 활성화 함수가 원점을 중심으로 작동하게 되어 학습 제한됨

#### 🔍 다층 신경망 구조
하나 이상의 은닉층을 포함하며 비선형 문제(XOR)를 해결할 수 있음
- 정규화 기법: L1 및 L2 정규화, Dropout, Batch Normalization

층 깊이를증가하면 신경망의 표현력이 강화되지만, 기울기 소실 문제가 발생 가능함 → Residual Connection 기법 활용
- MLP의 출력층: 분류 문제에서는 Softmax, 회귀 문제에서는 선형 활성화 함수 사용
- MLP의 학습 과정: 손실 함수와 Optimizer(최적화 알고리즘)이 중요한 역할을 하며 Cross-Entropy 손실 함수와 Adam Optimizer가 주로 사용됨

#### 🔍 활성화 함수
- Sigmoid 함수
  1. 출력 범위를 (0,1)로 제한하여 확률적 해석 가능
  2. 기울기 소실 문제 발생 가능
  3. 이진 분류 문제에서 널리 사용됨
     
- ReLU 함수
  1. 입력이 0 이하면 0 출력, 양수일 때는 그대로 출력함
  2. sigmoid보다 학습이 빠르고 기울기 소실 문제 완화 가능
  3. 입력값이 0 이하일 경우는 Dying ReLU 문제 발생 → Leakly ReLU 함수
     
- Tanh 함수
  1. sigmoid와 유사하지만 출력 범위가 (-1,1)로 설정 → 중심이 0인 데이터 학습할 때 유리함
  2. 기울기 소실 문제 발생 가능

- 분류 문제에서는 Softmax, 회귀 문제에서는 Linear Activation function 사용 (은닉층이 아닌 출력층에서 사용)

<br>

## Chapter 3-3. 신경망 학습 프로세스

#### 🔍 순전파
신경망에서 입력 데이터가 각 층을 통과하며 출력을 생성하는 과정으로 신경망의 학습과 예측에서 핵심적인 단계
행렬 연산을 활용하여 최적화됨

```text
class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN, self).__init__() 
    self.fc1 = nn .Linear(2, 3) # 입력층(2) → 은닉층(3)
    self.fc2 = nn.Linear(3, 1) # 은닉층(3) → 출력층(1)
    
  def forward(self, X): 
    x = torch.relu(self.fc1(x)) # 활성화 함수 적용 
    x = torch.sigmoid(self.fc2(x))  # 확률 형태의 값이 출력됨
    return x

model = SimpleNN() 
sample_input = torch.tensor([1.0, 2.0]) 
output = model(sample_input) 
print(f "SimpleNN 출력: {output}")
```
→ 초기화된 가중치가 다르기 때문에 확률적으로 실행될 때마다 결과가 다를 수 있음

#### 🔍 역전파
<img width="100" height="50" alt="image" src="https://github.com/user-attachments/assets/e5d7b0fe-030a-4244-b8d3-e522971d90ee" />
<img width="150" height="40" alt="image" src="https://github.com/user-attachments/assets/56b432f9-3b35-4aa1-8a9d-540fda7e7ea0" />

손실 함수의 기울기를 출력층부터 입력층 방향으로 계산하는 방식으로 진행됨
각 층에서 가중치와 편향에 대한 편미분을 구하고, 이를 이용하여 가중치를 업데이트함
→ 파이토치에서는 Autograd 기능을 활용하여 자동으로 기울기 계산 가능

#### 🔍 손실 함수와 옵티마이저
1. MSE: 평균 제곱 오차
<img width="300" height="70" alt="image" src="https://github.com/user-attachments/assets/8f0ae438-261c-494c-bb5c-aa233689d235" />
2. Cross Entropy: 분류 문제에서 사용되는 손실 함수로 모델의 예측 확률과 실제 정답 간의 차이를 측정함
   <img width="550" height="70" alt="image" src="https://github.com/user-attachments/assets/4f667889-9e6c-4e44-a153-1a79f9f6e34a" />
3. SGD: 확률적 경사 하강법, 반복마다 무작위 샘플을 선택하여 가중치를 업데이트하는 방식(일부 샘플만을 이용해 학습을 진행함) → mini batch SGD도 자주 사용됨
4. Adam Optimizer: SGD의 단점을 보완하기 위해 개발된 옵티마이저, 일반적으로 a = 0.0001 사용
