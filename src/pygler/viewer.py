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

from gui import GlutWindow
from OpenGL import GL
from OpenGL.error import NoContext

from graphics.shader import Shader
from gui.viewcontroller import ViewController
from model import PyGLerModel,Geometry
from utils import CameraParams
import time

from pygler import FragmentShaderCode, VertexShaderCode

class PyGLer(object):

    def __init__(self, windowWidth=640,windowHeight=480, useFBO=False,cameraParams=CameraParams()):
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
        self._pointSize = 2
        self.controller = ViewController()
    
    @property
    def pointSize(self):
        return self._pointSize
    
    @pointSize.setter
    def pointSize(self,v):
        if self.started:
            raise BaseException("GL point size cannot be set after the viewer is started.")
        self._pointSize = v

    
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
        self._needsRedraw=True
                
    def on_idle(self,dt):
        if self.started and self._needsRedraw==True:
            self._needsRedraw=False
            self.window.redraw()

        if self._captureRequested is True:
            self.capturedTuple = self.captureRGBD()
            with self.actionCond:
                self._captureRequested=False
                self.actionCond.notifyAll()
        
        if self._model2Remove is not None:
            with self.lock:
                try:
                    self.models.remove(self._model2Remove)
                    self._model2Remove.cleanUp()
                    self._needsRedraw=True
                except ValueError:
                    print("No Model named ",self._model2Remove.name)
                    print(sys.exc_info())
                    
            with self.actionCond:
                self._model2Remove=None
                self.actionCond.notifyAll()
                        
                
        if self._stopRequested==True:
            if self.started and self.window!=None:
                try: 
                    self.window.stop()    
                except Exception:
                    e = sys.exc_info()[0]
                    print("Exception while stopping GLUT window: ",e)
                    print(sys.exc_info())
            with self.actionCond:
                self._stopRequested=False
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
            print("No OpenGL Context found. If you are calling from a different thread use the capture() method instead.")
            print(sys.exc_info())
        
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
            raise Exception("Failed to properly setup FBO. OpenGL returned FBO status: ",status)

        #unbind FBO
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,0)
        print("FBO initialized correctly.")
        
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
                self.window = GlutWindow((self.width,self.height), title="PyGLer", enableAlpha=not self.useFBO, pointSize=self._pointSize)
                self.window.event(self.on_resize)
                self.window.event(self.on_draw)
                self.window.event(self.on_idle)
                
                if self.useFBO is True:
                    self._initFBO()
                
                # Create the Shader program.
                self.shader = Shader(VertexShaderCode, FragmentShaderCode)
                
                projMat = self._cameraParams.projectionMat
                projMat = projMat.reshape(-1).tolist() 
            
                self.shader.bind()
                self.shader.uniform_matrixf("projM", projMat)

                self.controller.registerEvents(self.window)
                self.window.show()
                
            except Exception as e:
                self.window = None # failed to create window
                print("Exception in PyGLer run. Failed to initialize GlutWindow: \n",e)
                raise Exception("Failed to initialize GlutWindow.",e)
        
        try:
            self.started = True                
            self.window.start() # this call is blocking (calls the GLUTMainLoop)
                        
            # When stop is called we must release the resources
            print("Releasing Resourses")
            if self.useFBO is True:
                self._releaseFBO()
            
            for m in self.models:
                m.cleanUp()
            
            self.window=None
            
        except Exception:
            e = sys.exc_info()[0]
            print("Exception in GlUT event loop: ",e)
            print(sys.exc_info())
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
                raise Exception("Failed to remove model. There is already a model pending for removal.")
            self._model2Remove = model 
            while self._model2Remove!=None: # while model is not removed
                self.actionCond.wait()

            
    def removeAll(self):
        for m in self.models:
            self.removeModel(m)
                    
    def on_draw(self):
        if self.started is False:
            return
        
        view = self.controller.getViewM()
        self.shader.uniform_matrixf("viewM", view)
        red,green,blue,alpha = 1,1,1,1
        GL.glClearColor(red,green,blue,alpha)
        
        if self.useFBO is True:
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
        
        with self.lock:
            for m in self.models:
                if m.geometry.needsVAOUpdate is True:
                    m.geometry.updateVAO(self.shader)
                            
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            for m in self.models:
                self.shader.uniformf("singleColor",-1,-1,-1,1)
                m.draw(self.shader,self.controller.showMesh,self.controller.showFaces,self.controller.showNormals)


    def on_resize(self, width, height, x=0, y=0):
        self.width = width
        self.height = height
        GL.glViewport( 0, 0, width, height )
        self.redraw()
        
        
    @staticmethod
    def Convert2BGRD(rawRGBAXYZW, scale=1000.0, depthMin=2,depthMax=10000.0):
        rgba,xyzw = rawRGBAXYZW
        bgr = (rgba[:,:,2::-1]*255.0).astype(np.ubyte)
        dtmp = xyzw[:,:,3]
        dtmp[dtmp<depthMin] = 0
        dtmp[dtmp>depthMax] = 0
        
            
        depth = (dtmp*scale).astype(np.ushort)
        
        return (depth,bgr)


from utils import ComputeNormals
from pygler.utils import CreateAxisModel, CreateCubeModel

if __name__ == '__main__':
    print("Opening window")
    viewer = PyGLer(useFBO=True)
    
                                     
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
    
    
    m = PyGLerModel("Test",Geometry(vertices=testV,triangles=testF,autoScale=False, alpha=0.5))
    cube = CreateCubeModel("cube",scale=4.0,alpha=0.3)
     
    # tri = PyGLerModel.LoadObj("triceratops.obj",computeNormals=True)
    
    axis = CreateAxisModel(thickness=0.05)
    viewer.addModel(axis)
    # viewer.addModel(tri)
    viewer.addModel(cube)
    viewer.addModel(m)

    viewer.start()
     
#     while True:
#         depth,bgr = viewer.Convert2BGRD(viewer.capture())
#         image.show("Depth",depth,30)
    
    
    
    
    
    