import tensorrt as trt  
import torch
import numpy as np
import cv2
from collections import OrderedDict, namedtuple

class DetectMultiBackend(torch.nn.Module):
    def __init__(self, trtPath):      
        super().__init__()
        device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(trtPath, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())

        # 将本地变量赋值给类成员变量的方法
        self.__dict__.update(locals()) 

    def forward(self, img):
        batch_size = self.bindings['input_data'].shape[0]  # if dynamic, this is instead max batch size
        print(batch_size)
        s = self.bindings['input_data'].shape
        assert img.shape == s, f"input size {img.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['input_data'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]
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


