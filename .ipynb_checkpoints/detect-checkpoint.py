from PIL import Image
import torch,time
from torch.nn import functional as F 
import torch.nn as nn
from torchvision import transforms
from MyTorch2trt import load_trtModel


def main():
        
    #Step 0:查看torch版本、设置device
    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Step 1:准备数据集
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    #Step 2: 初始化网络
    trt_file = "Cat_Dog.pth"
    model = model = load_trtModel(trt_file)

    pic_path = "./img/cat.jpg"
    with torch.no_grad():   
        # 时间
        start_time = time.time()
        # 输入图片并修改大小，添加一个维度
        image = Image.open(pic_path)
        image = test_transform(image)
        image = image.to(device)
        image = image[None]
        # 推理
        output = model(image)
        output = torch.softmax(output, dim=1)
        # 推理结果处理
        lables_dic = {0:"cat",1:"dog"}
        result = torch.argmax(output[0]).item()
        acc = output[0,result].item()
        # 输出
        print("精度：",acc,"种类：",lables_dic[result],"time:",time.time()-start_time)
        
if __name__ == '__main__':
    main()