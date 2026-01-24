## Chapter 7-1. 트랜스포머 구조의 이해

#### 🔍 개념 정리
- 트랜스포머: 문장 전체를 한 번에 보면서 어떤 단어가 어떤 단어를 중요하게 봐야하는지 계산하는 모델
- Self Attention: 각 요소가 다른 모든 요소와 상호 작용해 각 요소의 중요도를 동적으로 결정함 <br>
  1. Query: 나와 관련 있는 단어는?
  2. Key: 나는 어떤 정보를 갖고 있는가? 
  3. Value: 실제로 전달할 내용은 무엇인가?
- Query와 Key를 이용해 raw attention score 구하기<br>

- 자기 집중(self-focus)와 문맥적 연관성(contextual links)을 동시에 학습
- Multi-head attention: self-attention을 여러 번 병렬로 수행해 다양한 표현으로 학습함
- Post wise Feed Forward Networks: 각 위치에서 독립적으로 적용되는 완전 연결 네트워크 <br>
  - 비선형 변환 수행
  - feed forward network를 각 위치에 개별적으로 적용함
- Encoder-Decoder: 입력 시퀀스를 받아 고차원적인 표현으로 인코딩하고 인코더의 출력을 받아 최종 출력 시퀀스를 생성
  
<br>

#### ❓ 보충 내용 정리
1. @ code 의미: 행렬 곱 (cf. *는 원소별 곱)
2. W_q, W_k, W_v가 전부 같아 보여도 randn로 생성했기 때문에 값이 전부 다름<br>
   처음만 랜덤으로 생성되고 학습이 되면 오차를 줄이도록 각자 다른 역할로 업데이트(미분 경로가 다름)

````python
W_q = torch.randn(self.d_model, self.d_model) 
W_k = torch.randn(self.d_model, self.d_model) 
W_v = torch.randn(self.d_model, self.d_model)

Q = embeddings @ W_q # Query: "누구와 관련있나?“ 
K = embeddings @ W_k # Key: "나의 특성은?" 
V = embeddings @ W_v # Value: " 나의 실제 정보”
````
3. 미분 경로가 다른 이유: Q랑 K가 먼저 곱해져 attention score를 만들고 value는 나중에 곱해짐 -> back loss 구할 때 미분 경로가 다름
````python
scores = Q @ K.T / np.sqrt(self.d_model)
````
4. X는 단어 임베딩(의미, 위치, 문맥 정보 등 전부 섞여 있음), W는 학습되는 투영 행렬(무엇을 중요하게 볼 지 결정)
<img width="300" height="50" alt="image" src="https://github.com/user-attachments/assets/1bcdb1ed-ae43-48f7-8046-e3a57fed1d33" /> <br>
5. Encoder-Decoder attention vs self-attention
   1) self-attention
      - 같은 sequence 안에서 참고
      - Q, K, V 모두 같은 sequence에서 나옴
   2) Encoder-decoder 
     - 서로 다른 sequence 참고
     - Q는 Decoder, Key&Value는 Encoder 내용을 참고


## Chapter 7-2. 사전 학습 모델 활용과 전이 학습 통합

#### 🔍 개념 정리
- 전이 학습: 대규모 데이터셋으로 학습된 사전 학습 모델의 지식을 새로운 문제에 전이해 비교적 적은 양의 데이터만으로 높은 성능을 낼 수 있도록 하는 기법
  1. Feature Extraction: 사전 학습 모델의 초기 레이어를 고정한 채 일부만 새롭게 학습하는 방식
  2. Fine-Tuning: 사전 학습 모델의 일부 또는 전체 파라미터를 새로운 데이터셋에 맞게 함께 학습시키는 방식
- Tokenizer: 입력 문장을 모델이 처리할 수 있는 토큰으로 변환함
<br>

## Chapter 7-3. 자연어 처리와 Vision Transformer 개요

#### 🔍 개념 정리
- Vision Tranformer: NLP 분야의 트랜스포머 아키텍처를 이미지 처리에 적용한 모델
- NLP와 ViT의 공통점
  1. 트래스포머 기반
  2. 포지셔널 인코딩: NLP에서는 단어의 순서, ViT에서는 이미지 패치의 위치 정보를 보존함
- NLP와 ViT의 차이점
  1. 입력 형태: 텍스트 시퀀스를 입력으로 받아 임베딩 벡터로 변환 / 이미지를 작은 패치로 분할해 각 패치를 벡터로 임베딩
  2. 모델 구조: 인코더 구조를 사용해 문맥 정보 학습 / 전체 이미지에 대한 분류 수행
  3. 활용 분야: 번역, 감성 분석, 요약 / 이미지 분류, 객체 검출 등






