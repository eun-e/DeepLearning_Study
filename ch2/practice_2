import torch

def main():
  tl = torch.tensor([1, 2, 3]) 
  t2 = torch.tensor([4, 5, 6]) 
  
  # cat: 주어진 차원을 따라 텐서를 이어 붙임 
  cat_result = torch.cat((tl , t2), dim=0) # shape: (6,) 
  print(f'cat_result: {cat_result}') 
  
  # stack: 새 차원을 만들어 텐서를 쌓음 
  stack_result = torch.stack((tl , t2), dim=0) # shape: (2, 3) 
  print(f'stack_result:\n {stack_result}')

  big_tensor = torch.arange(10)
  
  # chunk: 텐서를 여러 개로 나눌 때 개수(chunk) 기준으로 나눔
  chunk_result = torch.chunk(big_tensor, chunks=3)
  print(f'chunk_result: {chunk_result}')

if __name__ == "__main__":
    main()
