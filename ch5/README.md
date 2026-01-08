## Chapter 5-1. 합성곱 신경망의 기본 개념과 구조

#### 📝 개념 정리
- DNN: 입력층 - 여러 은닉층 - 출력층 (Fully Connected Layer만 사용) <br>
  └ 입력 데이터를 1차원 벡터로 펼쳐서(flatten) 사용함
- CNN: Convolution layer, Pooling layer 거치고 마지막에 Fully Connected layer 사용 <br>
  └ 필터로 일부 영역만 보고 같은 필터를 전체에 적용해 가중치를 공유함 <br>
  └ 입력 이미지에서 객체 위치가 달라져도 동일한 특징을 감지할 수 있음

## Chapter 5-2. 합성곱 신경망의 주요 구성 요소

#### 📝 개념 정리
- Convolution layer: 입력 데이터에 필터를 적용하여 특징을 추출
- CNN의 필터는 밝기 변화가 
