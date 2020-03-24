# PyTorch_ONNX_TensorRT
A tutorial that show how could you build a TensorRT engine from a PyTorch Model with the help of ONNX. Please kindly star this project if you feel it helpful.


# Environment
0. Ubuntu 16.04 x86_64, CUDA 10.0
1. Python 3.5
2. [PyTorch](https://pytorch.org/get-started/locally/) 1.0 
3. TensorRT 5.0 (If you are using Jetson TX2, TensorRT will be already there if you have installed the jetpack)  
3.1 Download [TensorRT](https://developer.nvidia.com/tensorrt) (You should pick up the right package that matches your environment)  
3.2 Debian installation
```
  $ sudo dpkg -i nv-tensorrt-repo-ubuntu1x04-cudax.x-trt5.x.x.x-ga-yyyymmdd_1-1_amd64.deb # The downloaeded file
  $ sudo apt-key add /var/nv-tensorrt-repo-cudax.x-trt5.x.x.x-gayyyymmdd/7fa2af80.pub
  $ sudo apt-get update
  $ sudo apt-get install tensorrt
  
  $ sudo apt-get install python3-libnvinfer
```
To verify the installation of TensorRT
`$ dpkg -l | grep TensorRT`
You should see similar things like
```
  ii  graphsurgeon-tf	5.1.5-1+cuda10.1	amd64	GraphSurgeon for TensorRT package
  ii  libnvinfer-dev	5.1.5-1+cuda10.1	amd64	TensorRT development libraries and headers
  ii  libnvinfer-samples	5.1.5-1+cuda10.1	amd64	TensorRT samples and documentation
  ii  libnvinfer5		5.1.5-1+cuda10.1	amd64	TensorRT runtime libraries
  ii  python-libnvinfer	5.1.5-1+cuda10.1	amd64	Python bindings for TensorRT
  ii  python-libnvinfer-dev	5.1.5-1+cuda10.1	amd64	Python development package for TensorRT
  ii  python3-libnvinfer	5.1.5-1+cuda10.1	amd64	Python 3 bindings for TensorRT
  ii  python3-libnvinfer-dev	5.1.5-1+cuda10.1	amd64	Python 3 development package for TensorRT
  ii  tensorrt	5.1.5.x-1+cuda10.1	amd64	Meta package of TensorRT
  ii  uff-converter-tf	5.1.5-1+cuda10.1	amd64	UFF converter for TensorRT package
```

3.2 Install PyCuda (This will support TensorRT)
  ```
   $ pip3 install pycuda 
  ```
If you get problems with pip, please try

 ```
 $ sudo apt-get install python3-pycuda #(Install for /usr/bin/python3)
 ```
For full details, please check the [TensorRT-Installtation Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)


# Usage
Please check the file 'pytorch_onnx_trt.ipynb'

# News:
Int8 demo updated! Just  
```python
python3 trt_int8_demo.py
```  

You will see output like
 
  >Function forward_onnx called!  
  >graph(%input : Float(32, 3, 128, 128),  
        %1 : Float(16, 3, 3, 3),  
        %2 : Float(16),  
        %3 : Float(64, 16, 5, 5),  
        %4 : Float(64),  
        %5 : Float(10, 64),  
        %6 : Float(10)):  
    %7 : Float(32, 16, 126, 126) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1]](%input, %1, %2), scope: Conv2d  
    %8 : Float(32, 16, 126, 126) = onnx::Relu(%7), scope: ReLU  
    %9 : Float(32, 16, 124, 124) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1]](%8), scope: MaxPool2d  
    %10 : Float(32, 64, 120, 120) = onnx::Conv[dilations=[1, 1], group=1,   kernel_shape=[5, 5], pads=[0, 0, 0, 0], strides=[1, 1]](%9, %3, %4), scope: Conv2d  
    %11 : Float(32, 64, 120, 120) = onnx::Relu(%10), scope: ReLU  
    %12 : Float(32, 64, 1, 1) = onnx::GlobalAveragePool(%11), scope:   AdaptiveAvgPool2d  
    %13 : Float(32, 64) = onnx::Flatten[axis=1](%12)  
    %output : Float(32, 10) = onnx::Gemm[alpha=1, beta=1, transB=1](%13, %5, %6), scope: Linear  
    return (%output)
  Int8 mode enabled
  Loading ONNX file from path model_128.onnx...  
  Beginning ONNX file parsing  
  Completed parsing of ONNX file  
  Building an engine from file model_128.onnx; this may take a while...  
  Completed creating the engine  
  Loading ONNX file from path model_128.onnx...  
  Beginning ONNX file parsing  
  Completed parsing of ONNX file  
  Building an engine from file model_128.onnx; this may take a while...  
  Completed creating the engine  
  Loading ONNX file from path model_128.onnx...  
  Beginning ONNX file parsing  
  Completed parsing of ONNX file  
  Building an engine from file model_128.onnx; this may take a while...  
  Completed creating the engine  
  Toal time used by engine_int8: 0.0009500550794171857  
  Toal time used by engine_fp16: 0.001466430104649938  
  Toal time used by engine: 0.002231682623709525  

This output is run by Jetson Xavier.  
Please be noted that int8 mode is only supported by specific GPU modules, e.g. Jetson Xavier , Tesla P4, etc. 

TensorRT 6 and 7 have been released. Although the tutorial is run with TensorRT 5.0, it should be also compatible with TensorRT 6 and 7.



# Contact
Cai, Rizhao    
Email: rizhao.cai@gmail.com
