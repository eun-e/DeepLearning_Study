## Chapter 6-1. 순환 신경망의 기초

#### 🔍 개념 정리
- 시계열 데이터: 시간 흐름에 따라 순차적으로 수집된 데이터로, 주식 가격, 음성 신호, 텍스트 등이 해당함 <br>
  └ 관측값이 시간적인 의존성을 가짐
- RNN(순환 신경망): 이전 단계의 정보를 기억하는 순환 구조를 가짐 (은닉층의 뉴런이 자기 자신과 연결) <br>
  <img width="298" height="52" alt="image" src="https://github.com/user-attachments/assets/871c09b3-6f27-45df-9504-ce71a3016019" /> <img width="177" height="56" alt="image" src="https://github.com/user-attachments/assets/49448e73-07d5-4cfc-ab28-74e05b34fc6e" />
  

#### ❓헷갈리는 내용 정리
1. RNN은 왜 기울기 소실과 기울기 폭발이 심각한 문제일까? <br>
   RNN에서 Whh는 시점에 관계없이 항상 고정된 값을 가짐, h(t-1)이랑 h(t)만 변함 <br>
   <img width="205" height="60" alt="image" src="https://github.com/user-attachments/assets/df0b9fad-d047-4ddb-bf19-f79bbc5a0357" /> <br>
   Chain rule에 의해 가중치가 거듭적으로 곱해지기 때문에 W<1이면 기울기 소실, W가 크면 기울기 폭발이 일어남
2. 재현성을 위한 시드 설정 - 특정 숫자를 바탕으로 무적위처럼 보이는 수를 설정<br>
   torch.manual_seed(42): Pytorch 내에서 가중치 초기화 등에 쓰이는 난수를 고정 <br>
   np.random.seed(42): numpy 라이브러리를 사용해 데이터를 섞거나 생성할 때 쓰이는 난수를 고정
<br>

## Chapter 6-2. 고급 순환 신경망 아키텍처

#### 🔍 개념 정리
- LSTM: 순환 신경망의 기울기 소실 문제를 해결하기 위해 제안된 아키텍처 (셀 상태와 게이트 메커니즘)
- 주요 게이트
  1. 망각 게이트: 셀 상태에서 어떤 정보를 버릴지 결정함
  2. 입력 게이트: 새로운 정보 중 어떤 것을 저장할지 결정함
  3. 출력 게이트: 셀 상태에서 어떤 정보를 출력할지 결정
- 기울기 소실 해결: 역전파 시 가중치를 연쇄적으로 곱하는 대신, 덧셈 연산 위주로 정보가 전달되기 때문
<br>
- GRU: LSTM의 간소화 버전<br>
  └ 업데이트 게이트와 리셋 게이트만 사용, 별도의 셀 상태 없이 은닉 상태만 사용
- BRNN: 시퀀스를 정방향과 역방향으로 처리하는 구조
<br>

## Chapter 6-3. 파이토치를 이용한 순환 신경망 구현

#### 🔍 개념 정리


  
