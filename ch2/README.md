## Chapter 2-1. 텐서의 개념과 생성


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
<br>


## Chapter 2-2. 텐서의 기본 연산
#### 🔍 개념 정리
- 브로드캐스팅: 서로 다른 크기의 텐서 간 연산할 수 있도록 하는 규칙 
```text
a = torch.tensor([1,2,3])
b = torch.tensor([10])
a+b = tensor([11, 12, 13]) # b가 자동으로 인덱스가 확장됨
```
- 인덱싱/슬라이싱: tensor2d[행 인덱스, 열 인덱스] 형태로 특정 위치의 원소를 얻을 수 있다 -> 콜론을 사용하면 여러 행이나 열 동시에 선택 가능
- 불리언 마스킹: mask = tensor2d > 50처럼 조건 부여하면 True/False가 담긴 마스크 텐서가 생성된다
- permute: 3차원 이상 텐서에서 차원 순서를 자유롭게 변경한다
- squeeze: 크기가 1인 차원을 제거한다
- unsqueeze: 특정 위치에 차원을 추가한다


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
