# GLRenderer.jl
High FPS rendering. Supports Depth, RGB, and RGB+Texture rendering.

## Setup 
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

### MCS
If you would like to use MCS directly with this project, please install
```shell
source venv/bin/activate

python -m pip install wheel
cd ${MCS_DIRECTORY}
python -m pip install -r requirements.txt
cd -
python -m pip install machine-common-sense
```

### Notebook
In order to use Jupyter notebooks with this project, please use:
```shell
source venv/bin/activate

PYTHON=$(which python) julia --project=@.
julia> ENV["PYTHON"]
julia> import Pkg; Pkg.build("PyCall")
Ctrl+D

PYTHON=$(which python) PYCALL_JL_RUNTIME_PYTHON=$(which python) jupyter-notebook
```
This activates the project's virtual Python environment, rebuilds PyCall with it and then launches the notebook while pointing to it.
