'''
Created on Mar 24, 2014

PyOpenGL viewer.

PyGLer is a simple mesh viewer using PyOpenGL to render (textured) triangle meshes.
PyGLer is based on Glumpy, some classed and modules are taken from Glumpy source


@author: padeler
'''
import threading
import sys

import numpy as np

from gui.glutwindow import GlutWindow
from OpenGL import GL
from OpenGL.error import NoContext

from graphics.shader import Shader
from gui.viewcontroller import ViewController
from model import PyGLerModel
from utils import CameraParams
import time


class PyGLer(object):
    
    
    vertexCode = """#version 330
            
                //uniform float end_fog;
                uniform vec4 singleColor;
                uniform mat4 projM;
                uniform mat4 viewM;
                uniform mat4 modelM;
                
                in vec4 position;
                in vec4 color;
                out vec4 vcolor;
                out vec4 vposition;
                void main() {
                    float fog; // amount of fog to apply
                    float fog_coord; // distance for fog calculation...

                    gl_Position = projM * viewM * modelM * position;
                    vposition = gl_Position;
                    if(singleColor.x==-1)
                    {
                        vcolor = color;
                    }
                    else
                    {
                        vcolor = singleColor;
                    }
                }"""
            
                    
    fragmentCode = """#version 330
            in vec4 vcolor;
            in vec4 vposition;
            out vec4 fragColor;
            out vec4 fragPos;
            void main() {
                    fragColor = vcolor;
                    fragPos = vposition;
                }"""
    
    
    def __init__(self, windowWidth=640,windowHeight=480, useFBO=False,cameraParams=CameraParams(), initViewM=np.eye(4,dtype=np.float32)):
        self.width=windowWidth
        self.height=windowHeight

        self._cameraParams = cameraParams
        
        self.models = []
        self.window = None
        self.lock = threading.Lock()
        self.actionCond = threading.Condition()
        self.useFBO = useFBO
        self.started=False
        
        self._needsRedraw=False
        self._captureRequested=False
        self._stopRequested=False
        self._model2Remove=None
        self.fbo = None
        self.renderBuffers = None

        self.controller = ViewController(initialViewM=initViewM)
    
    
    def stop(self):
        ''' 
        Releases any OpenGL resources held by this Viewer
        '''
        with self.actionCond:
            self._stopRequested = True
            while self._stopRequested and self.started:
                self.actionCond.wait(0.1)           
        
        
        
        
    def __del__(self):
        self.stop()

    
    def redraw(self):
        with self.lock:
            if self.window!=None:
                self._needsRedraw=True
                
    def on_idle(self,dt):
        if self._needsRedraw==True:
            self.window.redraw()
            self._needsRedraw=False

        if self._captureRequested==True:
            self.capturedTuple = self.captureRGBD()
            with self.actionCond:
                self._captureRequested=False
                self.actionCond.notifyAll()
        
        if self._model2Remove!=None:
            with self.lock:
                try:
                    self.models.remove(self._model2Remove)
                    self._model2Remove.cleanUp()
                    self._needsRedraw=True
                except ValueError:
                    print "No Model named ",self._model2Remove.name
                    print sys.exc_info()
                    
            with self.actionCond:
                self._model2Remove=None
                self.actionCond.notifyAll()
                        
                
        if self._stopRequested==True:
            if self.started and self.window!=None:
                try: 
                    self.window.stop()    
                except Exception:
                    e = sys.exc_info()[0]
                    print "Exception while stopping GLUT window: ",e
                    print sys.exc_info()
            with self.actionCond:
                self._captureRequested=False
                self.actionCond.notifyAll()

                
                
    def captureRGBD(self):
        '''
        XXX DO NOT USE THIS METHOD. Use the capture() instead. 
        This method is called from the GLUT thread.
        Captures a RGBD frame.
        '''
        try:
            if self.useFBO:
                res = []
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,self.fbo)
                for att in self.renderBufferAt:
                    # Read depth from the OpenGL buffer
                    GL.glReadBuffer(att)    
                    glBuffer = GL.glReadPixels(0, 0, self.fboWidth, self.fboHeight, GL.GL_RGBA, GL.GL_FLOAT)
                    img = np.flipud(np.array(glBuffer, np.float32).reshape(self.fboHeight,self.fboWidth,4))
                    res.append(img)
                                    
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
                return res
            
            # no fbo, just return RGBA from default frame buffer.
            glBuffer = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
            img = np.flipud(np.array(glBuffer, np.float32).reshape(self.height,self.width,4))
            return [img,None,None]

        except NoContext:
            print "No OpenGL Context found. If you are calling from a different thread use the capture() method instead."
            print sys.exc_info()
        
    def capture(self):
        '''
        Capture a raw frame and return it to host memory.
        Use PyGLer.Convert2BGRD to convert the captured images to 
        the typical depth and BGR pair a la kinect.
        '''
        with self.actionCond:
            self._captureRequested=True
            while self._captureRequested: # while there is no screenshot yet
                self.actionCond.wait()
            
            return self.capturedTuple
    
    
    def _releaseFBO(self):
        if self.renderBuffers!=None:
            GL.glDeleteRenderbuffers(len(self.renderBuffers),self.renderBuffers)
            self.renderBuffers=None
        if self.fbo!=None:
            GL.glDeleteFramebuffers(1,self.fbo)
            self.fbo = None
            
    
    def _initFBO(self):
        self.fboWidth = self.width
        self.fboHeight = self.height

        # Create three Render buffers. One for color one for XYZW one for depth_component
        self.renderBuffers = GL.glGenRenderbuffers(3)

        # initialize storage for the render buffers
        for i in range(2):
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,self.renderBuffers[i])
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER,GL.GL_RGBA32F,self.fboWidth,self.fboHeight)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,0)

        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,self.renderBuffers[2])
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER,GL.GL_DEPTH_COMPONENT,self.fboWidth,self.fboHeight)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,0)
        
            
        self.fbo = GL.glGenFramebuffers(1)
        # bind buffer and populate with RenderBuffers
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,self.fbo)

        self.renderBufferAt = [GL.GL_COLOR_ATTACHMENT0,GL.GL_COLOR_ATTACHMENT1]

        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,self.renderBufferAt[0],GL.GL_RENDERBUFFER,self.renderBuffers[0])
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,self.renderBufferAt[1],GL.GL_RENDERBUFFER,self.renderBuffers[1])
        
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,GL.GL_DEPTH_ATTACHMENT,GL.GL_RENDERBUFFER,self.renderBuffers[2])
        
        # make sure that the new FBO  is complete
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise StandardError("Failed to properly setup FBO. OpenGL returned FBO status: ",status)

        #unbind FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
        
        print "FBO initialized correctly."
        
        
#         # allocate linear memory to copy the rendering output from FBO
#         self.PBO = GL.glGenBuffers(3)
#     
#         for i in range(3):
#             GL.glBindBuffer( GL.GL_PIXEL_PACK_BUFFER, self.PBO[i] )
#             GL.glBufferData( GL.GL_PIXEL_PACK_BUFFER, self.fboWidth * self.fboHeight * 4 * 4 , None, GL.GL_STREAM_COPY )
#             GL.glBindBuffer( GL.GL_PIXEL_PACK_BUFFER, 0 )

    
    def run(self):
        with self.lock:
            try:
                # Initialize the GUI Window (GlutWindow)
                self.window = GlutWindow((self.width,self.height), title="PyGLer")
                self.window.event(self.on_resize)
                self.window.event(self.on_draw)
                self.window.event(self.on_idle)
                
                if self.useFBO==True:
                    self._initFBO()
                
#                 Create the Shader program.
#                 vertexCode = open("vertex.glsl","r").read()
#                 fragmentCode = open("fragment.glsl","r").read()
                self.shader = Shader(self.vertexCode, self.fragmentCode)
                
                projMat = self._cameraParams.projectionMat
                projMat = projMat.reshape(-1).tolist() 
            
                self.shader.bind()
                self.shader.uniform_matrixf("projM", projMat)

                self.controller.registerEvents(self.window)
                self.window.show()
                
            except Exception,e:
                self.window = None # failed to create window
                print "Exception in PyGLer run. Failed to initialize GlutWindow: ",e
                raise StandardError("Failed to initialize GlutWindow.",e)
        
        try:
            self.started = True                
            self.window.start() # this call is blocking (calls the GLUTMainLoop)
            
            # When stop is called we must release the resources
            print "Releasing Resourses"
            if self.useFBO==True:
                self._releaseFBO()
            
            for m in self.models:
                m.cleanUp()
            
            self.window=None
            
        except Exception:
            e = sys.exc_info()[0]
            print "Exception in GlUT event loop: ",e
            print sys.exc_info()
        finally:
            self.started = False
            # Release resources
    
    def start(self):

        t = threading.Thread(target = self.run)
        t.start()
            
        while not self.started:
            time.sleep(1.0)
    

    
    def addModel(self,model):
        with self.lock:
            self.models.append(model)
            self._needsRedraw=True
            
    def removeModel(self,model):
        '''
        Removes the model from the viewer. 
        The model removal is handled by the viewer main thread. 
        '''
        with self.actionCond:
            if self._model2Remove!=None:
                raise StandardError("Failed to remove model. There is already a model pending for removal.")
            self._model2Remove = model 
            while self._model2Remove!=None: # while model is not removed
                self.actionCond.wait()

            
    def removeAll(self):
        for m in self.models:
            self.removeModel(m)
                    
    def on_draw(self):
        if self.started==False:
            return
        
        view = self.controller.getViewM()
        self.shader.uniform_matrixf("viewM", view)
        red,green,blue,alpha = 1,1,1,1
        GL.glClearColor(red,green,blue,alpha)
        
        
        if self.useFBO==True:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,self.fbo)
            GL.glDrawBuffers(2,self.renderBufferAt)
            self._draw() # Draw on the FBO
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
            
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo);
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0); # read from FBO color0
             
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0);
            GL.glDrawBuffers((GL.GL_BACK,)) # write to window frame buffer color0
             
            GL.glBlitFramebuffer(0,0,self.fboWidth,self.fboHeight,0,0,self.width,self.height,GL.GL_COLOR_BUFFER_BIT,GL.GL_LINEAR)
        else:
            self._draw()
        
    def _draw(self):
        
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        with self.lock:
            for m in self.models:# TODO: Create model VAO on_idle
                if m.needsVAOUpdate==True:  
                    m.updateVAO(self.shader)
                    
                self.shader.uniformf("singleColor",-1,-1,-1,1)
                m.draw(self.shader,self.controller.showMesh,self.controller.showFaces,self.controller.showNormals)


    def on_resize(self, width, height, x=0, y=0):
        self.width = width
        self.height = height
        GL.glViewport( 0, 0, width, height )
        
        
    @staticmethod
    def Convert2BGRD(rawRGBAXYZW, scale=1000.0):
        rgba,xyzw = rawRGBAXYZW
        bgr = (rgba[:,:,2::-1]*255.0).astype(np.ubyte)
        dtmp = xyzw[:,:,3]
        dmin = dtmp.min()
        if dmin!=0: # invalid depth is 0 for Kinect RGBD frames
            dtmp[dtmp==dmin]=0

        
        depth = (dtmp*scale).astype(np.ushort)
        
        return (depth,bgr)

        



from utils import ComputeNormals

if __name__ == '__main__':
    print "Opening window"
    viewer = PyGLer()
    
    s = 0.5                                                                
    cubeV = np.array([-s, s, s,
                   s, s, s,
                   s,-s, s,
                  -s,-s, s,
                  -s,-s,-s,
                  -s, s,-s,
                   s, s,-s,
                   s,-s,-s],dtype='f').reshape(-1,3)
    
    cubeF = np.array([0,1,2, 2,3,0, # front
                      2,3,4, 4,7,2, # bottom
                      0,5,4, 4,3,0, # left
                      7,2,1, 1,6,7, # right
                      5,6,7, 7,4,5, # back
                      0,5,6, 6,1,0 ],dtype=np.uint32).reshape(-1,3)
                                     
    testV = np.array( [
            [  0, 1, 0], #triangle
            [ -1,-1, 0],
            [  1,-1, 0],
            
            [  2,-2, 0], # parallelepiped 1
            [  4,-2, 0],
            [  4, 1, 0],
            [  2,-2, 0],
            [  4, 1, 0],
            [  2, 1, 0],
            
            [  2, 1,-2], # square2
            [  4, 1,-2],
            [  4, 1, 0],
            [  2, 1,-2],
            [  4, 1, 0],
            [  2, 1, 0]
            
        ],'f')
    
    testF = np.array([#0,1,2, # triangle
                      
                      3,4,5, # parallelepiped
                      6,7,8,
                      
                      11,10,9, # square
                      14,13,12
                      
                      ],np.uint32).reshape(-1,3)
    
    
    m = PyGLerModel("Test",testV,triangles=testF, normals = ComputeNormals(testV,testF), normalScale=0.2, autoScale=False)
    cube = PyGLerModel("Cube", vertices = cubeV, triangles=cubeF)
    
    tri = PyGLerModel.LoadObj("triceratops.obj",computeNormals=True)
    viewer.addModel(tri)
    viewer.addModel(cube)
    viewer.addModel(m)

    viewer.start()
