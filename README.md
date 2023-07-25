# PyGLer
Python OpenGL viewer

PyGLer is a simple mesh viewer using PyOpenGL to render (colored) triangle meshes and point clouds.


## Requirements

- glut: Make sure GLUT is installed. I.e in ubuntu use the freeglut package


## Installation

Preparation:
```bash
# Change in the root of the project
# create python env
python3 -m venv pyenv 
source pyenv/bin/activate
```

Dependencies and path:
```bash
pip install --upgrade pip
pip install -r requirements.txt
# set pythonpath
export PYTHONPATH=$PWD:$PYTHONPATH
```


## Running the viewer

```bash
python pygler/viewer.py view --model media/triceratops.obj --show-axis
```


## Viewer Controls

### Mouse
- Left: Rotate
- Right + move: Scale object
- Wheel: Scale
- Middle click: Translate object

### Keyboard

- Arrow Keys: Move camera
- w a s d: Move camera
- q e: Roll Camera
- z x: Pitch Camera
- r: Reset view
- 1: Toggle mesh
- 2: Toggle faces
- 3: Toggle normals
- i: Print OpenGL info

