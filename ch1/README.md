#### 🔍 개념 정리
- 텐서플로: 정적 계산 그래프
  ```text
  import tensorflow as tf
  
  x = tf.Variable(2.0, name='x')

  with tf.GradientTape as tape:
    y = x**2 + 3*x + 4
    grad = tape.gradient(y,x)
  ```
  
- 파이토치: 동적 계산 그래프 (실행 중 생성, 연산 후 소멸)
  ```text
  import torch

  x = torch.tensor(2.0, requies_grad=True)
  y = x**2 + 3*x + 4

  y.backward()
  print(x.grad) # x.grad는 동적으로 계산된 미분값을 출력
  ```

- 텐서: 수학에서 사용하는 다차원 배열 개념을 확장한 것으로, 딥러닝과 머신러닝에서 데이터를 표현하는 기본 단위
