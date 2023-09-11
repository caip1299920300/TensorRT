import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

''' 想要实现校准器的功能，需继承TensorRT提供的四个校准器类中的一个，然后重写父校准器的几个方法：
    get_batch_size: 用于获取batch的大小
    get_batch: 用于获取一个batch的数据
    read_calibration_cache: 用于从文件中读取校准表
    write_calibration_cache: 用于把校准表从内存中写入文件中
'''
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.d_input = cuda.mem_alloc(self.data_loader.calibration_data.nbytes)
        self.cache_file = cache_file
        data_loader.reset()

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        batch = self.data_loader.next_batch()
        if not batch.size:
            return None
        # 把校准数据从CPU搬运到GPU中
        cuda.memcpy_htod(self.d_input, batch)

        return [self.d_input]

    def read_calibration_cache(self):
        # 如果校准表文件存在则直接从其中读取校准表
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        # 如果进行了校准，则把校准表写入文件中以便下次使用
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()