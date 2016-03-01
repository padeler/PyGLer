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

in vec4 position;
in vec4 color;
varying out vec4 vcolor;
varying out vec4 vposition;
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

in vec4 vcolor;
in vec4 vposition;
varying out vec4 fragColor;
varying out vec4 fragPos;

void main() {
    fragColor = vcolor;
    fragPos = vposition;
}

"""


