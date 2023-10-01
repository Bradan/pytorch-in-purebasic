# Using PyTorch in PureBasic for Inference

This repository contains a very small example how to use PyTorch (CPU) in PureBasic for inference. It works on Arch Linux 64bit, but should also work on any other system with some adaptions. This is far from perfect, maybe an OpenVino interface would be better suited or any ONNX parser. 

I've chosen the [MIT license](LICENSE) for this code, but you likely need a different dataset if you really want to use this project. The here mentioned MNIST dataset is not part of the project and has to be downloaded separately, you might choose a different one.

## How to Build

1. Download the dependencies
   ```bash
   ./download_dependencies.sh
   ```
2. Build the simplified PyTorch wrapper for PureBasic
   ```bash
   mkdir build
   pushd build
   cmake ..
   make -j 8
   popd
   ```
3. Set up a python virtual environment with PyTorch and activate it
   ```bash
   python3 -m venv ./venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Train a model and copy the last epoch to the *pb* folder as *test.pt* (this will download the MNIST dataset using PyTorch's routines).
   ```bash
   pushd python
   python3 ./train.py
   cp epoch_2.pt ../pb/test.pt
   popd
   ```
5. Copy all libraries to the *pb* folder (I don't know how to add a library path for PureBasic Import statements).
   ```bash
   cp build/libPBTorch.a pb/
   cp libs/*/lib/*.{so,a}* pb/
   ```
6. Compile and run the PureBasic source code *pb/test.pb*.
   ![Screenshot](screenshot.png?raw=true "Screenshot")