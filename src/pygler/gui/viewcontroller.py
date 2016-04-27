'''
Created on Mar 28, 2014

@author: padeler
'''

import numpy as np
from pygler.gui import key, mouse

from trackball import Trackball
from OpenGL import GL

class ViewController(object):
    '''
    Creates a view matrix based on mouse and keyboard input.
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self._initialViewM = np.eye(4,dtype=np.float32)
        self._viewM = np.copy(self._initialViewM)
        
        self.trackball = Trackball()
        self.keyDict = { 
                        ord('r') : self.reset,
                        key.UP:    self.up,
                        key.DOWN:  self.down,
                        key.LEFT:  self.left,
                        key.RIGHT: self.right,
                        ord('w'):  self.forward,
                        ord('s'):  self.backward,
                        ord('a'):  self.left,
                        ord('d'):  self.right,

                        ord('q'):  self.rollLeft,
                        ord('e'):  self.rollRight,

                        ord('z'):  self.pitchUp,
                        ord('x'):  self.pitchDown,

                        ord('i'):  self.info,
                        
                        ord('1'):  self.toggleMesh,
                        ord('2'):  self.toggleFaces,
                        ord('3'):  self.toggleNormals,
                                                
                        
                        }
        self.translateMult = -1.0
        self.dragMult = -0.01
        self.zoomMult = 0.01
        self._zoom = 1.0
        
        self.showMesh = True
        self.showFaces = False
        self.showNormals = False
        
            
    def info(self):
        print "OpenGL Version: ",GL.glGetString(GL.GL_VERSION)
        print "OpenGL Vendor: ",GL.glGetString(GL.GL_VENDOR)
        print "OpenGL Renderer: ",GL.glGetString(GL.GL_RENDERER)
        print "OpenGL GLSL Version: ",GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)
        print "OpenGL Extensions: \n",GL.glGetString(GL.GL_EXTENSIONS).split()
        print "FBO Max color attachments: ",GL.glGetIntegerv(GL.GL_MAX_COLOR_ATTACHMENTS)
#         print "FBO Max Width: ",GL.glGetInteger(GL.GL_MAX_FRAMEBUFFER_WIDTH)
#         print "FBO Max Height: ",GL.glGetInteger(GL.GL_MAX_FRAMEBUFFER_HEIGHT)
#         print "FBO Max Samples: ",GL.glGetInteger(GL.GL_MAX_FRAMEBUFFER_SAMPLES)
#         print "FBO Max Layers: ",GL.glGetInteger(GL.GL_MAX_FRAMEBUFFER_LAYERS)
        


    def unmapped(self):
#         self.width = 640
#         self.height =480
#         
#         # Read image from the OpenGL buffer
#         buffer = ( GL.GLfloat * (self.width*self.height) )(0)
#         GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, buffer)
#         depth = np.flipud(np.array(buffer, np.float32).reshape(self.height,self.width))
# 
#         # Read depth from the OpenGL buffer
#         buffer = ( GL.GLubyte * (3*self.width*self.height) )(0)
#         GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, buffer)
#         bgr = np.flipud(np.array(buffer, np.uint8).reshape(self.height,self.width,3))
# 
#            
#         image.show("Depth", depth)
#         image.show("BGR", bgr,30)
#         print "Unmapped key pressed."
        pass

        
                
        
    def toggleMesh(self):
        self.showMesh = not self.showMesh
        self.parentWindow.redraw()

    def toggleFaces(self):
        self.showFaces = not self.showFaces
        self.parentWindow.redraw()

    def toggleNormals(self):
        self.showNormals = not self.showNormals
        self.parentWindow.redraw()
    
    def on_key_press(self,symbol, modifiers):
        self.keyDict.get(symbol, self.unmapped)();
        self.parentWindow.redraw()
    
    def on_mouse_scroll(self,dx, dy,s0,s1):
        self._zoom += self.zoomMult * s1
        
        if self._zoom<self.zoomMult:# do not allow negative zoom
            self._zoom = self.zoomMult
            
#         print "Zoom ",self._zoom
        self.parentWindow.redraw()

    def on_mouse_drag(self, x, y, dx, dy, buttons):
#         print "Mouse Drag ",x,y," => ",dx,dy," Buttons: ",mouse.buttons_string(buttons) 

        if buttons&mouse.MIDDLE:
            self.translate(-dx*self.dragMult, dy*self.dragMult, 0)
        elif buttons&mouse.RIGHT:
            self.translate(0, 0, dy*self.dragMult)
        else:
            self.trackball.dragTo(x, y, -dx, dy)
            
        self.parentWindow.redraw()
        
        pass
    
    def on_mouse_release(self,x,y,buttons):
#         print "Mouse Release ",x,y," => Buttons: ",buttons
        pass
    
    def rollLeft(self):
        self.rotate(0,-1)
        
    def rollRight(self):
        self.rotate(0,+1)
        
        
    def pitchUp(self):
        self.rotate(-1,0)
        
    def pitchDown(self):
        self.rotate(1,0)

    def rotate(self,dTheta, dPhi):
        theta,phi = self.trackball.getOrientation()
        phi += dPhi
        theta +=dTheta
        self.trackball.setOrientation(theta,phi)

    def registerEvents(self,window):
        window.event(self.on_key_press)
        window.event(self.on_mouse_drag)
        window.event(self.on_mouse_scroll)
        window.event(self.on_mouse_release)
        self.parentWindow=window
        
    def reset(self):
        print "Reset View."
        self._viewM = self._initialViewM.copy()
        self._zoom = 1.0
        self.trackball.setOrientation(0, 0)
            
    def up(self):
#         print "Up."
        dy = -1.0 * self.translateMult 
        self.translate(0, dy, 0)

    def down(self):
#         print "Down."
        dy = 1.0 * self.translateMult 
        self.translate(0, dy, 0)

    def left(self):
#         print "Left."
        dx = -1.0 * self.translateMult 
        self.translate(dx, 0, 0)

    def right(self):
#         print "Right."
        dx = 1.0 * self.translateMult 
        self.translate(dx, 0, 0)
        
    def forward(self):
#         print "Forward."
        dz = 1.0 * self.translateMult 
        self.translate(0, 0, dz)

    def backward(self):
#         print "Backward."
        dz = -1.0 * self.translateMult 
        self.translate(0, 0, dz)


    def translate(self,dx,dy,dz):
        tr = np.eye(4,dtype=np.float32)
        tr[0:3,3] = np.array([dx,dy,dz],dtype=np.float32)
        self._viewM = tr.dot(self._viewM)
    
    
    def getViewM(self):
        '''
        Return the view matrix in PyOpenGL usable format (flat list of floats, column major)
        ''' 
        scaleM = np.diag([self._zoom,self._zoom,self._zoom,1.0])
        viewM = self._viewM.dot(self.trackball.getRotMat().dot(scaleM))
        return viewM.transpose().reshape(-1).tolist()


    def setCameraPosition(self,trXYZ,rotQuat=[1,0,0,0],scale=1):
        '''
        set the camera orientation. This will not set the reset location.
        :param trXYZ:
        :param rotQuat: in the form of: x,y,z,w
        :param scale:
        :return:
        '''
        tr = np.eye(4,dtype=np.float32)
        tr[0:3,3] = np.array(trXYZ,dtype=np.float32)
        self._viewM = tr
        self._zoom = scale
        self.trackball.setRotation(rotQuat)


    
