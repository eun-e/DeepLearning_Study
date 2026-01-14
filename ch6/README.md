## Chapter 6-1. 순환 신경망의 기초

#### 🔍 개념 정리
- 시계열 데이터: 시간 흐름에 따라 순차적으로 수집된 데이터로, 주식 가격, 음성 신호, 텍스트 등이 해당함 <br>
  └ 관측값이 시간적인 의존성을 가짐
- RNN(순환 신경망): 이전 단계의 정보를 기억하는 순환 구조를 가짐 (은닉층의 뉴런이 자기 자신과 연결) <br>
  <img width="298" height="52" alt="image" src="https://github.com/user-attachments/assets/871c09b3-6f27-45df-9504-ce71a3016019" /> <img width="177" height="56" alt="image" src="https://github.com/user-attachments/assets/49448e73-07d5-4cfc-ab28-74e05b34fc6e" />
-



#### ❓헷갈리는 내용 정리
1. RNN은 왜 기울기 소실과 기울기 폭발이 심각한 문제일까? <br>
   RNN에서 Whh는 시점에 관계없이 항상 고정된 값을 가짐, h(t-1)이랑 h(t)만 변함 <br>
   <img width="205" height="60" alt="image" src="https://github.com/user-attachments/assets/df0b9fad-d047-4ddb-bf19-f79bbc5a0357" /> <br>
   Chain rule에 의해 가중치가 거듭적으로 곱해지기 때문에 W<1이면 기울기 소실, W가 크면 기울기 폭발이 일어남
