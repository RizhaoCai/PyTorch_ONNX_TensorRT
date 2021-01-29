import torch
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt # TensorRT 7

import pdb
import torch.nn as nn
import os

import numpy as np
import torch

TRT_LOGGER = trt.Logger()


class Net(nn.Module):
    def __init__(self, num_clasess=2):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 8, 3, 2)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(8,num_clasess)
    def forward(self, x):batchsize
        
        out = self.pool(self.conv(x))
        out = out.view(-1, 8)
        out = self.fc(out)
        return out

def to_onnx():
    # Export to onnx
    global input_shape

    net = Net()

    net.eval()

    dummy_input = torch.zeros([2,]+input_shape)
    out = net(dummy_input)
    print(out.shape)
    onnx_model_path = 'net.onnx'
    torch.onnx.export(net, dummy_input, onnx_model_path, verbose=True, input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}},
        opset_version=11)
    print('Exported the model to ', onnx_model_path)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        #pdb.set_trace()
        size = trt.volume(engine.get_binding_shape(binding)[1:]) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes) # Only bytes, no need for size
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    
    # Run inference.
    success_flag = context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle) # Bug

    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream

    return [out.host for out in outputs]


def get_engine(max_batch_size=1,onnx_file_path="", engine_file_path="",fp16_mode=False,int8_mode=False,save_engine=False,test_set_fname=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    global input_shape
    def build_engine(max_batch_size):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1000#1 <<  # 1GB
            builder.max_batch_size = max_batch_size
            #pdb.set_trace()
            builder.fp16_mode = fp16_mode
            builder.int8_mode = int8_mode
            config = builder.create_builder_config()
            profile = builder.create_optimization_profile()
            

            if int8_mode:
                exit("Not implemented")

            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parsing_succeed = parser.parse(model.read())
                
                #if not parsing_succeed:
                #    exit('Failed to parse the ONNX model')
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            
            # Static input
        
            profile.set_shape('input', [1,]+ input_shape, [2,]+ input_shape, [max_batch_size,]+ input_shape)
            config.add_optimization_profile(profile)
  
            engine = builder.build_engine(network, config=config)
          

            if not engine:
                exit('Failed to build the engine')
            
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                    print("Completed creating Engine")
            return engine
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size)

def test_dynamic_shape():
    #engine, context,  h_input, d_input, h_output, d_output, stream = onnx_2_tensorrt.main()
    global max_batch_size, input_shape, num_clasess
    onnx_file_path = 'net.onnx'
    fp16_mode = False
    int8_mode = False
    
    
    engine_file_path = "net_fp16_{}_int8_{}_bs_{}.trt".format(fp16_mode,int8_mode,max_batch_size)

    print("Building Engine")

    calibration_stream = None


    engine = get_engine(max_batch_size,onnx_file_path,engine_file_path,fp16_mode=fp16_mode,int8_mode=int8_mode,test_set_fname=None)


    inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings
    
    context = engine.create_execution_context()

    for batchsize in range(1, max_batch_size+1 ):
        x = np.ones([batchsize,]+input_shape).astype(np.float32)
            
        inputs[0].host = x.data
        context.set_binding_shape(0, [batchsize,]+input_shape)
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
        print('\nBatchSize='+str(batchsize)+'\n', trt_outputs[0].reshape(-1, num_clasess), )
        """
            You will see the some output like (the value may be different):
                BatchSize=1
                [[-0.4005382   0.37080884]
                [ 0.          0.        ]
                [ 0.          0.        ]
                [ 0.          0.        ]]

                BatchSize=2
                [[-0.4005382   0.37080884]
                [-0.4005382   0.37080884]
                [ 0.          0.        ]
                [ 0.          0.        ]]

                BatchSize=3
                [[-0.4005382   0.37080884]
                [-0.4005382   0.37080884]
                [-0.4005382   0.37080884]
                [ 0.          0.        ]]

                BatchSize=4
                [[-0.4005382   0.37080884]
                [-0.4005382   0.37080884]
                [-0.4005382   0.37080884]
                [-0.4005382   0.37080884]]
            
            ! The output_shape depends how you allocate the max batch size and memory. 
            ! Zeros will be filled to where the batch dimensions > batchsize   
        """
    pdb.set_trace()
if __name__ == '__main__':
    
    input_shape = [3,32,32]
    num_clasess = 2
    max_batch_size = 4
    to_onnx()
    test_dynamic_shape()
    
