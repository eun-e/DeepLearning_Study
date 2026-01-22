## Chapter 7-1. 트랜스포머 구조의 이해

#### 🔍 개념 정리
- 트랜스포머: 문장 전체를 한 번에 보면서 어떤 단어가 어떤 단어를 중요하게 봐야하는지 계산하는 모델
- Self Attention: 각 요소가 다른 모든 요소와 상호 작용해 각 요소의 중요도를 동적으로 결정함 <br>
  1. Query: 나와 관련 있는 단어는?
  2. Key: 나는 어떤 정보를 갖고 있는가? 
  3. Value: 실제로 전달할 내용은 무엇인가?
- Query와 Key를 이용해 raw attention score 구하기<br>

- 자기 집중(self-focus)와 문맥적 연관성(contextual links)을 동시에 학습

#### ❓ 보충 내용 정리
1. @ code 의미: 행렬 곱 (cf. *는 원소별 곱)
2. 
````python
W_q = torch.randn(self.d_model, self.d_model) 
W_k = torch.randn(self.d_model, self.d_model) 
W_v = torch.randn(self.d_model, self.d_model)

Q = embeddings @ W_q # Query: "누구와 관련있나?“ 
K = embeddings @ W_k # Key: "나의 특성은?" 
V = embeddings @ W_v # Value: " 나의 실제 정보”
````
W_q, W_k, W_v가 전부 같아 보여도 randn로 생성했기 때문에 값이 전부 다름
