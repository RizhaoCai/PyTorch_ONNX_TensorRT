import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator):
    def __init__(self, input_layers, stream, cache_file='calibration_cache.bin'):
        trt.IInt8EntropyCalibrator.__init__(self)       
        self.input_layers = input_layers
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, bindings, names):
        batch = self.stream.next_batch()
        if not batch.size:   
            return None
        
        cuda.memcpy_htod(self.d_input, batch)
        for i in self.input_layers[0]:
            assert names[0] != i

        bindings[0] = int(self.d_input)
        return bindings

    def read_calibration_cache(self, length):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, ptr, size):
        #cache = ctypes.c_char_p(int(ptr))
        value = ctypes.pythonapi.PyCapsule_GetPointer(ptr, None)

        '''
        # TODO: If the calibration is read from cache 'calibration_cache.bin', it will raise bugs
        #       Will solve this in the future.
        with open(self.cache_file, 'wb') as f:
            #f.write(cache.value)
            f.write(value)
        '''
        return None
    

class ImageBatchStreamDemo():
    def __init__(self,dataset, transform, batch_size, img_size, max_batches=10):
        '''
            For calibiration, you need to implement your 'next_batch' and 'reset' functions
        '''
        self.transform = transform
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.dataset = dataset
          
        # self.calibration_data = np.zeros((batch_size, 3, 800, 250), dtype=np.float32)
        self.calibration_data = np.zeros((batch_size,)+ img_size, dtype=np.float32) # This is a data holder for the calibration
        self.batch_count = 0
        
        
    def reset(self):
      self.batch_count = 0
      
    def next_batch(self):
        """
        Return a batch of data every time called
        """
        #self.max_batches = 2
        if self.batch_count < self.max_batches:
            i = self.batch_count
            for i in range(self.batch_size):
                # You should implement your own data pipeline for writing the calibration_data
                
                x = self.dataset[i]   
                if self.transform:
                    x = self.transform(x) 
                
                self.calibration_data[i] = x.data
            self.batch_count += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32) 
        else:
            return np.array([])
