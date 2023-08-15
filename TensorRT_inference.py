import tensorrt as trt  
import torch
import numpy as np
import cv2
from collections import OrderedDict, namedtuple

class DetectMultiBackend(torch.nn.Module):
    def __init__(self, trtPath):      
        super().__init__()
        device = torch.device('cuda:0') # 设置设备为CUDA设备
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))  # 定义命名元组Binding
        logger = trt.Logger(trt.Logger.INFO)  # 创建TensorRT日志记录器
        with open(trtPath, 'rb') as f, trt.Runtime(logger) as runtime: 
            model = runtime.deserialize_cuda_engine(f.read()) # 从TensorRT引擎文件反序列化模型
        context = model.create_execution_context() # 创建TensorRT执行上下文
        bindings = OrderedDict() # 创建有序字典存储绑定数据
        output_names = [] # 存储输出名称的列表
        fp16 = False  # 是否使用FP16数据类型，默认为False
        dynamic = False # 是否为动态形状，默认为False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)  # 获取绑定的名称
            dtype = trt.nptype(model.get_binding_dtype(i))   # 获取绑定的数据类型
            if model.binding_is_input(i):  # 判断是否为输入绑定
                if -1 in tuple(model.get_binding_shape(i)):  # 判断是否为动态形状
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2])) # 设置绑定的动态形状
                if dtype == np.float16:
                    fp16 = True
            else:  # 输出绑定
                output_names.append(name) # 添加输出名称到列表中
            shape = tuple(context.get_binding_shape(i)) # 获取绑定的形状
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device) # 创建一个空的Tensor作为绑定的数据
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))  # 将绑定信息添加到bindings字典中
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items()) # 创建有序字典存储绑定的内存地址

        # 将本地变量赋值给类成员变量的方法
        self.__dict__.update(locals()) 

    def forward(self, img):
        batch_size = self.bindings['input_data'].shape[0]  # 获取绑定数据的批量大小（若为动态形状，则为最大批量大小）
        s = self.bindings['input_data'].shape # 获取绑定数据的形状
        assert img.shape == s, f"input size {img.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}" # 断言输入数据的形状与模型形状匹配
        self.binding_addrs['input_data'] = int(img.data_ptr()) # 更新输入数据的内存地址
        self.context.execute_v2(list(self.binding_addrs.values()))  # 执行TensorRT推理
        y = [self.bindings[x].data for x in sorted(self.output_names)] # 获取排序后的输出数据
        return y

if __name__ == "__main__":
    im = cv2.imread("test.jpg")
    im = cv2.resize(im,(768,640))
    im = torch.from_numpy(im).to(torch.device('cuda:0'))
    im = im.float() 
    im /= 255
    im = im[None] 
    im = im.permute(0, 3, 1, 2) 

    inference = DetectMultiBackend("best.engine")
    y = inference(im)
    print(len(y),[i.shape for i in y])
    print(y)


