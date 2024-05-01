from PIL import Image
import torch,time
from torch.nn import functional as F 
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from MyTorch2trt import save_engine,load_trtModel

# 添加一个softmax层
class Softmax_model(nn.Module):
    def __init__(self):
        super(Softmax_model,self).__init__()
        self.model = models.resnet18(num_classes=2)  
        # 加载训练好的权重
        trained_weight = torch.load('./resnet18_Cat_Dog.pth')
        self.model.load_state_dict(trained_weight)
    
    def forward(self, x):
        x = self.model(x)
        return x

def main():
    model = Softmax_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 保存trt
    save_engine(model,test_size=[224,224],model_name="Cat_Dog",workspace=10,fp16_mode=True)

if __name__ == '__main__':
    main()