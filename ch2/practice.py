import torch

def main():
        # 특수 함수로 텐서 생성
        zeros_tensor = torch.zeros((3,3))
        ones_tensor = torch.ones((2,4))
        full_tensor = torch.full((2,2), 7) 
        
        arange_tensor = torch.arange(0,10, step=2)  
        linspace_tensor = torch.linspace(0,1, steps=5)  # 중요 포인트!! step은 간격, steps는 개수 의미

        print("zeros_tensor:\n", zeros_tensor)
        print("ones_tensor:\n", ones_tensor)
        print("full_tensor:\n", full_tensor)
        print("arange_tensor:\n", arange_tensor)
        print("linspace_tensor:\n", linspace_tensor)

        
        # 무작위 텐서 생성
        rand_tensor = torch.rand((3,3))
        randn_tensor = torch.randn((2,2))        # 평균이 0, 표준편차가 1인 정규 분포
        
        print("rand_tensor:\n", rand_tensor)
        print("randn_tensor:\n", randn_tensor)

if __name__ == "__main__":
    main()
