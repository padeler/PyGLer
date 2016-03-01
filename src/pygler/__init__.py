'''
Python - OpenGL - Viewer (PyGLer)

'''



VertexShaderCode = \
"""
#version 130

uniform vec4 singleColor;
uniform mat4 projM;
uniform mat4 viewM;
uniform mat4 modelM;

in vec4 position;
in vec4 color;
out vec4 vcolor;
out vec4 vposition;
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
#version 130

in vec4 vcolor;
in vec4 vposition;
out vec4 fragColor;
out vec4 fragPos;

void main() {
    fragColor = vcolor;
    fragPos = vposition;
}

"""


