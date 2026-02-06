## Chpater 9-1. 자연어 처리 모델 구현 및 학습

#### 🔍 개념 정리
< 자연어 처리 >
- 컴퓨터가 인간의 언어를 이해하고 생성할 수 있도록 하는 분야
- 기계 번역, 감성 분석, 질의응답, 음성 인식 및 합성 등에 활용됨
- 텍스트 전처리 기법
    1. 토큰화: 문장을 단어 또는 하위 단위(형태소)로 분리함, 규칙은 언어마다 다르며 영어는 공백과 구두점으로 단어 구분
    2. 정규화: 모든 문자를 소문자로 변환하거나 불필요한 구두점과 특수문자를 제거해 일관된 형태로 정리함 <br>
       → ex. don't 를 do not으로 변환함
    3. 불용어 제거: 분석에 크게 기여하지 않는 단어를 제거함
       → a나 the는 문법적으로는 필요한 단어이지만 내용상 중요도가 낮은 단어
    4. 패딩: 문장 길이를 일정하게 맞추기 위해 짧은 문장은 0 등으로 채움
    5. 워드 임베딩: 단어를 실수 벡터로 변환하여 단어 간 의미적 유사성을 반영하는 기법
       - 원-핫 인코딩: 표현하려는 단어의 해당 위치는 1, 나머지는 모두 0으로 표현함
       - 워드투벡터
         - 중심 단어의 입력 임베딩과 모든 단어의 출력 임베딩 간 내적을 통해 출력 주변 단어 확률 예측
         - 입력 임베딩과 출력 임베딩은 무작위 수에서 시작하고 학습하면서 변함 
         - CBOW는 주변 단어들로 중심 단어 예측, Skip-gram은 중심 단어로 주변 단어 예측함                 
       - 글로브: 전체 말뭉치의 통계적 정보를 이용해 임베딩 학습함, 단어 ij가 같은 문맥에서 얼마나 자주 등장했는지
       - 문맥 기반 임베딩: 같은 단어라도 문맥에 따라 서로 다른 벡터 얻
- 모델링 방법
  1. 순환 신경망(RNN): 장기 의존성 문제에 의해 긴 문장에서는 초기 정보가 잘 전달되지 않음
  2. LSTM: 오래된 정보는 기억하고 불필요한 정보는 잊는 메커니즘
     - 셀 상태: 내부 메모리를 유지함
     - 게이트: 정보를 저장할지 버릴지 결정함 (입력, 출력, 망각 게이트)
       ````python
     class LSTMGateExplainer:
      def explain_gates(self, important_info, current_input, memory):
        forget_decision = self.forget_gate(memory, current_input)
        remember_decision = self.input_gate(current_input)
        output_decision = self.output_gate(memeory, current_input)
    
      def forget_gate(self, memory, current):
        if "날씨" in current or "점심" in current:
          return "일상적인 정보"
        return "중요 정보 유지"

      def input_gate(self, current):
        if "비밀번호" in current or "중요" in current:
          return "핵심 정보"
        return "일반 정보"
    
      def output_gate(self, memory, current):
        if "금고" in current:
          return "비밀번호 관련 정보"
        return "일반 응답"
      ```` 
  3. GRU: LSTM을 간소화해 셀 상태를 따로 두지 않고 은닉 상태에 통합해 구조를 단순화함 (업데이트 게이트, 리셋 게이트)
  4. Transformer: self-attention 메커니즘, 인코더-디코더 구조, multi-head attention
<br>

❓추가 학습 내용
1. BERT
2. GPT

<br>

## 9-2. LSTM과 감성 분석기 실
    
