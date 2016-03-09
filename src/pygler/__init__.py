'''
Python - OpenGL - Viewer (PyGLer)

'''



VertexShaderCode = \
"""
#version 100

uniform vec4 singleColor;
uniform mat4 projM;
uniform mat4 viewM;
uniform mat4 modelM;

attribute vec4 position;
attribute vec4 color;

varying vec4 vcolor;
varying vec4 vposition;

void main() {

    gl_Position = projM * viewM * modelM * position;
    vposition = gl_Position;
    if(singleColor.x==-1.0)
    {
        vcolor = color;
    }
    else
    {
        vcolor = singleColor;
    }
}

"""


FragmentShaderCode = \
"""
#version 100

precision highp float;

varying vec4 vcolor;
//varying vec4 vposition;

//out vec4 fragPos;

void main() {
    gl_FragColor = vcolor;
 //   fragPos = vposition;
}

"""


