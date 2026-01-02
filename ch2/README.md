<img width="654" height="119" alt="image" src="https://github.com/user-attachments/assets/ccde0035-586c-4677-be6e-05aa661aa7ab" />## Chapter 2-1. 텐서의 개념과 생성

#### 🔍 텐서의 핵심 속성
1. 모양: 각 차원의 크기를 의미하며 [3](1차원), [2,3](2차원) 등으로 나타남
2. 자료형: 데이터 유형을 의미하며 int64(정수), float32(실수), bool(불린) 등으로 나타남
3. 장치: 텐서가 저장된 위치를 의미하며 'cpu', 'cuda:0'으로 나타남

#### 🔍 특수 함수로 텐서 생성할 때 주의할 점
- arange_tensor에서 사용되는 step은 간격 의미
- linspace_tensor에서 사용되는 steps는 개수 의미(0과 1을 포함해 사이에 들어가는 개수)

#### 📝 practice_1.py 실행 결과

```text
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])

tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]])

tensor([[7, 7],
        [7, 7]])

tensor([0, 2, 4, 6, 8])

tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

#### 🔍 데이터 타입
1. int_tensor.float(): 정수형 텐서를 32비트 부동 소수점 자료형으로 변환
2. int_tensor.long(): 텐서를 64비트 정수 자료형으로 변환
3. int_tensor.double(): 텐서를 64비트 부동 소수점 자료형으로 변환
<br>


## Chapter 2-2. 텐서의 기본 연산
#### 🔍 개념 정리
- 브로드캐스팅: 서로 다른 크기의 텐서 간 연산할 수 있도록 하는 규칙(차원이 다름)
```text
a = torch.tensor([1,2,3])
b = torch.tensor([10])
a+b = tensor([11, 12, 13]) # b가 자동으로 인덱스가 확장됨
```
- 인덱싱/슬라이싱: tensor이[행 인덱스, 열 인덱스] 형태로 특정 위치의 원소를 얻을 수 있다 <br>
  → 콜론을 사용하면 여러 행이나 열 동시에 선택 가능 <br>
  → tensor[1:, :]은 1행부터 끝까지라는 뜻(행 슬라이스)
  
- 불리언 마스킹: mask = tensor > 50처럼 조건 부여하면 T/F가 담긴 마스크 텐서가 생성된다 <br>
  → mask만 print하면 True/False 값이 담긴 마스크 텐서가 생성됨 <br>
  → tensor[mask]이런식으로 print하면 조건에 맞는 원소들만 나열됨
- 텐서 합치기
  1. torch.cat: 특정 차원을 기준으로 텐서를 이어 붙인다
  2. torch.stack: 새로운 차원을 만들어 텐서를 쌓는다
- 텐서 나누기
  1. torch.split: 특정 크기만큼씩 텐서를 잘라서 리스트로 변환한다, n개씩
  2. torch.chunk: 텐서를 여러 개로 나눌 때 개수 기준으로 나눈다, n덩어리
- 차원 조작하기
  1. permute: 3차원 이상 텐서에서 차원 순서를 자유롭게 변경한다
  2. squeeze: 크기가 1인 차원을 제거한다
  3. unsqueeze: 특정 위치에 차원을 추가한다
   transpose: 행과 열의 위치를 바꾼다다


#### 📝 practice_2.py 실행 결과

```text
cat_result: tensor([1, 2, 3, 4, 5, 6])
stack_result:tensor([[1, 2, 3],
                     [4, 5, 6]])
chunk_result: (tensor([0, 1, 2, 3]), tensor([4, 5, 6, 7]), tensor([8, 9]))
```
<br>


## Chapter 2-3. 텐서와 NumPy 배열 비교
#### 🔍 개념 정리
numpy 배열 덧셈은 결과가 [7 9 11 15] 이런식으로 표현되고, pytorch 텐서 덧셈은 [7, 9, 11, 15] 이런식으로 표현됨

- 자동 미분(AutoGrad)기능
  requires_grad=True 옵션을 사용하면 텐서의 연산 기록을 추적하여 역전파를 자동 계산함
- .to(device): 텐서를 원하는 디바이스(CPU or GPU)로 이동할 수 있음음
