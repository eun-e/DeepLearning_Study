## Chapter 4-1. 파이토치를 이용한 모델 학습 과정 이해

딥러닝 모델 학습은 입력 데이터를 모델에 전달하여 예측 결과를 얻고, 그 결과와 실제 정답 간 오차를 줄이기 위해 모델의 파라미터를 지속해서 업데이트하는 과정이다.

#### 🔍 개념 정리
- 순전파: 입력 데이터를 모델의 각 계층을 거쳐 최종 예측 결과를 도출하는 과정 <br>
  └ 계층별 연산에서 모델의 각 계층은 선형 변환이 먼저 수행됨 <br>
    └ torch.mm(x, W)는 행렬 곱셈을 수행하여 입력 x와 가중치 행렬 W를 곱함 <br>
  └ 이어 비선형 활성화 함수를 적용해 데이터의 복잡한 패턴을 학습함
- 역전파: 손실 함수로부터 각 계층의 파라미터에 대한 기울기를 계산함 <br>
  └ 기울기가 너무 작아지면 가중치 업데이트가 제대로 이루어지지 않아 학습이 정체됨 (기울기 소실)
- 손실함수: criterion = nn.MSELoss(), loss = criterion(model_out, y_true) 이런식으로 코드짜기 <br>
  └ 학습 단계마다 optimizer.zero_grad()를 호출해서 기울기를 초기화해야함!!
- 자동 미분의 활용
  1. requires_grad = True로 설정된 텐서에 대해 모든 연산 기록을 자동으로 저장함
  2. loss.backward()를 호출하면 저장된 연산 기록을 기반으로 파라미터의 기울기를 자동 계산함
- 옵티마이저: 역전파 과정에서 계산된 기울기를 기반으로, 각 파라미터를 업데이트하는 알고리즘

#### 📝 practice_2.py 
````text
Epoch 20, Loss: 0.0824
Epoch 40, Loss: 0.0350
Epoch 60, Loss: 0.0310
Epoch 80, Loss: 0.0275
Epoch 100, Loss: 0.0244
````

## Chapter 4-2. 데이터 전처리와 DataLoader 활용

#### 🔍 개념 정리
- 정규화: 입력 특성의 스케일을 맞춰 학습이 안정될 수 있게 함
- 데이터 증강: 학습 데이터의 다양성을 확보해 모델이 다양한상황에 대해 일반화할 수 있도록 도움 ex) transforms(회전)
- 데이터 분할: 학습, 검증, 테스트 데이터로 나눠 일반화 성능 강화

## Chapter 4-3. 모델 성능 평가와 개선

#### ❓헷갈렸던 내용들
- x의shape은 어떻게 표현될까 → [batch_size, feature_size] <br>
- 이미지의 feature size 표현법: (depth, height, width)인데 depth는 흑백 이미지면 1, 컬러면 RGB(3) <br>
  └ 투명도나 다른 특징이 더 포함되면 depth가 4, 5 이런식으로도 가능해짐 <br>
-
````text
# 1
self.fc1 = nn.Linear(input_size, hidden_size)
self.relu = nn.ReLU()
self.fc2 = nn.Linear(hidden_size, output_size)

#2
self.fc = nn.Sequential(
    nn.Linear(28*28, 128), 
    nn.ReLU(), 
    nn.Linear(128, 10)
    ) 

# 1과 2의 차이점: 같은 표현이지만 sequential은 입력이 순차적으로만 흐르는 모델을 간단히 정의할 때 사용함
# 1의 표현에서는 중간에 다른 연산이나 조건 등을 추가할 수 있음
````
- x = x.view(-1, 28*28) 표현에서 -1의 의미: 남은 차원의 크기를 PyTorch가 자동으로 계산하라는 뜻 <br>
  └ view(-1): 1차원으로 쭉 펴기 <br>
  └ view(n, -1): n을 고정한 상태로 나머지 차원의 크기를 (전체 원소 개수)//n으로 계산 <br>
- with torch.no_grad(): gradient 연산 옵션을 그만할 때 사용하는 함
