# 가중치 초기화 함수

def initialize_weights(model, init_type='kaiming'): 
  # 모델의 가중치 초기화 함수 
  for m in model.modules(): 
    if isinstance(m, nn.Conv2d): 
      if init_type == 'kaiming': 
      # He 초기화(ReLU와 함께 사용하기 좋음) 
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
      
      elif init_type == 'xavier': 
      # Xavier 초기화(시그모이드나 tanh과 함께 사용하기 좋음) 
        nn.init.xavier_normal_(m.weight) 
        
      elif init_type == 'normal': 
      # 정규 분포 초기화 
        nn.init.normal_(m.weight, mean=0, std=0.01) 
      if m.bias is not None: 
        nn.init.constant_(m.bias, 0) 
    
      elif isinstance(m, nn.BatchNorm2d): 
        nn.init.constant_(m.weight , 1) 
        nn.init.constant_(m.bias, 0) 
      elif isinstance(m, nn .Linear): 
        if init_type == 'kaiming': 
           nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
        elif init_type == 'xavier': 
           nn.init.xavier_normal_(m.weight) 
        elif init_type == 'normal': 
           nn.init.normal_(m.weight, mean=0, std=0.01) 
           nn.init.constant_(m.bias, 0) 
