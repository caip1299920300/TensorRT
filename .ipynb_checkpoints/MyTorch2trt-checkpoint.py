
from loguru import logger
import tensorrt as trt
from torch2trt import torch2trt,TRTModule
import torch

def save_engine(model, test_size=[192,256],workspace=10,batch=1,fp16_mode=True,model_name="model"):
    """
    将给定的PyTorch模型转换为TensorRT引擎，并保存模型的状态字典和引擎文件。

    参数:
    - model: 要转换的PyTorch模型。
    - test_size: 用于模型转换的输入尺寸，默认为[192, 256]。
    - workspace: TensorRT引擎转换时的最大工作空间大小（以字节为单位），默认为10（即2^10=1024*1024字节）。
    - batch: TensorRT引擎支持的最大批量大小，默认为1。
    - model_name: 保存的模型和引擎文件的名称前缀，默认为"model"。

    返回值:
    无。
    """
    # 创建模型输入的随机张量，并使用CUDA加速
    x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
    # 将PyTorch模型转换为TensorRT引擎，配置转换参数
    model_trt = torch2trt(
        model.cuda(),
        inputs=[x],
        fp16_mode=fp16_mode,    # 使用半精度模式
        log_level=trt.Logger.INFO,  # 设置日志级别
        max_workspace_size=(1 << workspace),  # 设置最大工作空间大小
        max_batch_size=batch,    # 设置最大批量大小
    )
    # 保存TensorRT模型的状态字典
    torch.save(model_trt.state_dict(), f"./{model_name}.pth")
    # 保存TensorRT引擎文件，供C++推理使用
    engine_file =  f"./{model_name}.engine"
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())    
    

def load_trtModel(trt_file="model.pth"):
    """
    加载经过TensorRT优化的模型。

    参数:
    - trt_file (str): TensorRT模型文件的路径，默认为"model.pth"。

    返回:
    - model: 加载好的TRT模型实例。
    """
    model = TRTModule()
    model.load_state_dict(torch.load(trt_file))  # 加载模型的状态字典
    # 预热模型（当前注释掉，可根据需要启用）
    # model(torch.rand(1,3,640,640).cuda())
    return model