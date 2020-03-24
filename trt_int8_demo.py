# Load a ONNX model
import numpy as np
import torch

from helpers import trt_helper
from helpers import trt_int8_calibration_helper as int8_helper
import time


class CNN(torch.nn.Module):
    def __init__(self, num_classes=10,):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Conv2d(3,16,3)
        self.layer2 = torch.nn.Conv2d(16,64,5)
        self.relu = torch.nn.ReLU()
        
        # TAKE CARE HERE
        # Ceil_mode must be False, because onnx eporter does NOT support ceil_mode=True
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=False) 
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1)) 
        
        self.fc = torch.nn.Linear(64,num_classes)
        self.batch_size_onnx = 0
        # FLAG for output ONNX model
        self.export_to_onnx_mode = False                  
      
    def forward_default(self, X_in):
        print("Function forward_default called! \n")
        x = self.layer1(X_in)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        
        # Such an operationt is not deterministic since it would depend on the input and therefore would result in errors
        length_of_fc_layer = x.size(1) 
        x = x.view(-1, length_of_fc_layer)
        
        x = self.fc(x)
        return x

    def forward_onnx(self, X_in):
        print("Function forward_onnx called! \n")
        x = self.layer1(X_in)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        assert self.batch_size_onnx > 0
        length_of_fc_layer = 64 # For exporting an onnx model that fit the TensorRT, processes here should be DETERMINISITC!
        x = x.view(self.batch_size_onnx, length_of_fc_layer) # 
        x = self.fc(x)
        return x

    def __call__(self, *args,**kargs):
        if self.export_to_onnx_mode:
            return self.forward_onnx(*args,**kargs)
        else:
            return self.forward_default(*args,**kargs)

def generate_onnx_model(onnx_model_path, img_size, batch_size):
    model = CNN(10)
    # This is for ONNX exporter to track all the operations inside the model
    batch_size_of_dummy_input = batch_size # Any size you want
    dummy_input = torch.zeros((batch_size_of_dummy_input,)+img_size, dtype=torch.float32)

    model.batch_size_onnx = batch_size_of_dummy_input
    model.export_to_onnx_mode = True
    input_names = [ "input" ]
    output_names = [ "output"] # Multiple inputs and outputs are supported
    with torch.no_grad():
        # If verbose is set to False. The information below won't displayed
        torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, input_names=input_names, output_names=output_names)

def main():
    # Prepare a dataset for Calibration
    batch_size = 32
    img_size = (3,128,128)
    onnx_model_path = 'model_128.onnx'
    generate_onnx_model(onnx_model_path, img_size, batch_size)

    dataset = np.random.rand(1000,*img_size).astype(np.float32)
    max_batch_for_calibartion = 5
    transform = None

    # Prepare a stream
    calibration_stream = int8_helper.ImageBatchStreamDemo(dataset, transform, max_batch_for_calibartion, img_size)

    engine_model_path = "engine_int8.trt"
    engine_int8 = trt_helper.get_engine(batch_size,onnx_model_path,engine_model_path, fp16_mode=False, int8_mode=True, calibration_stream=calibration_stream, save_engine=True)
    assert engine_int8, 'Broken engine'
    context_int8 = engine_int8.create_execution_context() 
    inputs_int8, outputs_int8, bindings_int8, stream_int8 = trt_helper.allocate_buffers(engine_int8)

    engine_model_path = "engine_int16.trt"
    engine_fp16 = trt_helper.get_engine(batch_size,onnx_model_path,engine_model_path, fp16_mode=True, int8_mode=False, save_engine=True)
    assert engine_fp16, 'Broken engine'
    context_fp16 = engine_fp16.create_execution_context() 
    inputs_fp16, outputs_fp16, bindings_fp16, stream_fp16 = trt_helper.allocate_buffers(engine_fp16)

    engine_model_path = "engine.trt"
    engine = trt_helper.get_engine(batch_size,onnx_model_path,engine_model_path, fp16_mode=False, int8_mode=False, save_engine=True)
    assert engine, 'Broken engine'
    context = engine.create_execution_context() 
    inputs, outputs, bindings, stream = trt_helper.allocate_buffers(engine)

    
    total_time_int8 = []
    total_time_fp16 = []
    total_time = []
    for i in range(1, dataset.shape[0]):
        x_input = dataset[i]
        inputs_int8[0].host = x_input.reshape(-1)

        tic_int8 = time.time()
        trt_helper.do_inference(context_int8, bindings=bindings_int8, inputs=inputs_int8, outputs=outputs_int8, stream=stream_int8)
        toc_int8 = time.time()
        total_time_int8.append(toc_int8 -tic_int8 )

        tic_fp16 = time.time()
        trt_helper.do_inference(context_fp16, bindings=bindings_fp16, inputs=inputs_fp16, outputs=outputs_fp16, stream=stream_fp16)
        toc_fp16 = time.time()
        total_time_fp16.append(toc_fp16 -tic_fp16 )

        tic = time.time()
        trt_helper.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        toc = time.time()
        total_time.append(toc -tic)
    
    print('Toal time used by engine_int8: {}'.format(np.mean(total_time_int8)))
    print('Toal time used by engine_fp16: {}'.format(np.mean(total_time_fp16)))
    print('Toal time used by engine: {}'.format(np.mean(total_time)))


if __name__ == '__main__':
    main()
