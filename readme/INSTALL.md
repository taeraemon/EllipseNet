# Installation

The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v1.7.0. NVIDIA GPUs are needed for both training and testing. We recommend you to use the virual environment provised by anaconda.

2. Install [COCOAPI](https://github.com/cocodataset/cocoapi)

    ~~~
    COCOAPI=/path/to/clone/cocoapi # Need to replace with your own path
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    ~~~

3. Clone this repo

    ~~~
    EllipseNet_ROOT=/path/to/clone/EllipseNet
    git clone https://git.openi.org.cn/capepoint/EllipseNet $EllipseNet_ROOT
    ~~~


4. Install the requirements

    ~~~bash
    cd $EllipseNet_ROOT
    pip install -r requirements.txt
    ~~~
    
5. Compile **Deformable convolutional** (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4))

    ~~~bash
    cd $EllipseNet_ROOT/src/lib/models/networks/DCNv2
    export TORCH_CUDA_ARCH_LIST=7.5 # May need for hardware related issue
    rm -rf build
    ./make.sh
    ~~~

6. Compile **Rotated_IOU related operation** 

    ```bash
    cd $EllipseNet_ROOT/src/lib/models/Rotated_IoU/cuda_op
    python setup.py install
    ```

