# GLRenderer.jl
High FPS rendering. Supports Depth, RGB, and RGB+Texture rendering.

## Setup

### Dependency Setup
Please run the following commands in order to setup the dependencies for GLRenderer. Tested under Ubuntu 21.10.
```shell
cd GLRenderer.jl/src/renderer
sudo apt install -y cmake libassimp5 libassimp-dev nvidia-cuda-toolkit
python3 -m pip install -r requirements.txt
sudo python setup.py develop
```

### GLRenderer Setup
Please use the following commands to setup GLRenderer itself.
```
cd GLRenderer.jl
python -m venv venv
source venv/bin/activate
cd src/renderer
python setup.py develop
```

This package also depends on: [Geometry](https://github.com/probcomp/Geometry) which is a probcomp private repo.

## Usage

Refer to `test` for examples of each of the types of rendering.

## Attribution & Licensing
The code in `src/renderer/pybind11` belongs to the [pybind11 project](https://github.com/pybind/pybind11) and is provided under a 3-clause BSD license we provide along with the code in the directory (see `src/renderer/pybind11/LICENSE`).
