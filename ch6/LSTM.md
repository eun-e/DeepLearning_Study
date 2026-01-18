## 보충 정리

### 📌 State란? 값(정보)
1. Cell state: 실제로 기억되는 정보, 장기 기억 저장소, 값이 누적되어 유지됨
2. Hidden state: 현재 시점에서 밖으로 보이는 상태, 다음 시점 계산에 이용됨
<br>

### 📌 Gate란? 조절자
1. Forget gate: 이전 기억을 얼마나 유지할지(버릴지) 비율을 결정함
2. Input gate: 새 정보 중 어떤 정보를 저장할 결정함
3. Output gate: 어떤 정보를 출력할 결정함
- 결정만 하고 값을 저장하지는 않음 <br>
- RNN은 무조건 이전 상태를 덮어쓰므로 기억 관리 기능이 없음 → 장기 의존성에 약한 이유 <br>
LSTM은 기억 관리자인 gate가 포함됨
<br>

### ⭐ Gate는 sigmoid, State는 tanh를 쓰는 이유
- Gate는 얼마나 통과시킬지 정하는 조절기이기 때문에 0~1 비율로 맞춰야함
- State는 내용을 저장함 (긍정/부정, 방향성 등)
<br>

### 📌 GRU와 차이점
1. Update gate: 과거 정보를 얼마나 유지할지 결정 → Forget+Input gate 역할 동시에 함
2. Reset gate: 이전 정보를 얼마나 무시할지 결정
3. cell state가 없는 대신 hidden state가 그 역할까지 대신함
<br>

### ⭐ LSTM shape
- 출력 텐서: (batch_size, sequence_length, hidden_sizeX2) <br>
  최종 은닉 상태 : (num_layersX2, batch_size, hidden_size) <br>
  최종 셀 상태: (num_layersX2, batch_size, hidden_size)
- 문장의 단어 개수가 T일 때, x = (batch_size, T, input_size) 이렇게 됨   
- 출력 텐서는 모든 시점의 hidden state를 다 모아둔 것이어서 중간 과정 결과임 (시간별 정보가 필요할 때 사용)
- 최종 은닉이랑 셀은 T 시점일 때 찍은거기 때문에 sequence length가 필요 없음
<br>

### ⭐ Window 생성
- 과거 N개를 한 묶음(윈도우)로 만드는 과정
````python
def create_windows(data, window_size, horizon=1):
  # data는 시계열 데이터, window_size는 과거를 몇 개 볼지, horizon은 몇 칸 뒤를 예측할지를 의미함
  X, y = [] , []
  # X는 입력(과거 묶음들), y는 정답(미래 값들)
  for i in range(len(data) - window size - horizon + 1): 
    X.append(data[i:(i + window_size)])
    # i번째부터 시작해서 window_size개 만큼 연속된 데이터 append하기
    y.append(data[i + window_size + horizon - 1]) 
  return np.array(X), np.array(y)
````
<img width="903" height="427" alt="image" src="https://github.com/user-attachments/assets/98d4c7cf-8b3b-4e88-a199-d344a3840b67" />



