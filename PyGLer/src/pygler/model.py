'''
Created on Mar 27, 2014

@author: padeler
'''
import numpy as np
from OpenGL import GL
from OpenGL.GL import ctypes


def EnsureDtype(**kwargs):
    for name,mat in kwargs.iteritems():
        if mat is not None and mat.dtype==np.float64:
            raise StandardError("Error. ",name," dtype should be float32.")


class Geometry(object):
    '''
    Geometry of a model. Contains vertices, faces, etc.
    It can be shared by one or more PyGLerModel instances. 
    '''
    
    VERTEX_VBO_INDEX=0
    COLOR_VBO_INDEX=1
    TRIANGLE_VBO_INDEX=2
    LINE_VBO_INDEX=3
    NORMAL_VBO_INDEX=4
    
    MESH_VAO=0
    LINE_VAO=1
    NORMAL_VAO=2
    
    def __init__(self, vertices=None, triangles=None, normals=None, colors=None, textureCoords=None, texture=None, autoScale=True, bgrColor=True, alpha=1.0, normalScale=0.05,glDrawMethod = GL.GL_STATIC_DRAW):
        self.VAO = None
        self.autoScale = autoScale
        self.vertices=None
        self.bgrColor = bgrColor
        self.normalScale = normalScale
        if alpha <0.0 or alpha > 1.0:
                raise StandardError("Alpha values must be in the range of [0.0,1.0]")
 
        self.alpha = alpha
        self.glDrawMethod = glDrawMethod
        

        self.update(vertices, triangles, normals, colors, textureCoords, texture)
    
    
    def update(self, vertices=None, triangles=None, normals=None, colors=None, textureCoords=None, texture=None):
        if vertices is not None: 
            EnsureDtype(vertices=vertices, triangles=triangles, 
                               normals=normals, colors=colors, textureCoords=textureCoords, texture=texture)
    
            # Ensure that the incoming data wont be changed before they are copied to the GPU.
            # Make a copy of everything in update()  
            if vertices is not None:
                vertices = np.copy(vertices)
            if triangles is not None:
                triangles = np.copy(triangles)
            if normals is not None:
                normals = np.copy(normals)
            if colors is not None:
                colors = np.copy(colors)
            if textureCoords is not None:
                textureCoords = np.copy(textureCoords)
            if texture is not None:
                texture = np.copy(texture)
                
                    
            if self.autoScale==True:
                # Scale vertices such that object is contained in [-1:+1,-1:+1,-1:+1]
                vmin, vmax =  vertices[:,0:3].min(), vertices[:,0:3].max()
                vertices[:,0:3] = 2*(vertices[:,0:3]-vmin)/(vmax-vmin) - 1
                    
            
            self.triangles = triangles
            self.textureCoords = textureCoords
            self.texture = texture
            self.normals = normals
            self.vertexCount = len(vertices)
            
            # if normals are supplied. Compute the vertices for the line segments that will represent them 
            # and add them to the end of the vertices list
    
            if normals is not None:
                if normals.shape!=vertices.shape:
                    raise StandardError("Normals array must have the same shape as the vertices array.")
                
                normalVertices = vertices + normals * self.normalScale 
    
                self.vertices = np.concatenate((vertices,normalVertices))
            else:
                self.vertices = vertices
    
            
            if colors is None:
                v = self.vertices.shape[0]
                colors = np.empty((v,4),dtype=np.float32)
                if self.normals is not None: # use the normals to color the surface if available
                    colors[:v/2,0:3] = (self.normals[:,0:3]+1)/2.0
                else: # fallback to the vertices coords.
                    colors[:,0:3] = (self.vertices[:,0:3]+1)/2.0
                    
                colors[:,3] = self.alpha # add alpha
                self.colors = colors                
            else:
                if colors.dtype!=np.float32:
                    colors = colors.astype(np.float32) / 255.0
                if len(colors.shape)==1:
                    colors = colors[np.newaxis,:]
                    
                if self.bgrColor: # convert to RGB from BGR
                    colors[:,:3] = colors[:,2::-1]
                    
                if colors.shape[0]==1: #single color
                    colorsTmp = np.empty((self.vertices.shape[0],3),dtype=np.float32)
                    colorsTmp[:,:3] = colors[0,:3]
                    colors = colorsTmp

                if colors.shape[1]==3 and self.alpha<1.0: # add alpha
                    colorsRGBA = np.empty((colors.shape[0],4),dtype=np.float32)
                    colorsRGBA[:,:3] = colors
                    colorsRGBA[:,3] = self.alpha
                    colors = colorsRGBA
                
                self.colors = colors
            

        self.needsVAOUpdate=True
    
    def getVertices(self):
        if self.normals is None:
            return np.copy(self.vertices)
        else:
            s = self.normals.shape[0]
            return np.copy(self.vertices[:s,:])
        
    def createVAO(self):
        # Create VAO for this Geometry
        self.VAO = GL.glGenVertexArrays(3)
        # Generate buffers to hold our vertices
        self.vertexBuffers = GL.glGenBuffers(5)
    
    def updateVAO(self,shader):
        
        if self.vertices is None or self.needsVAOUpdate==False: 
            return
        
        
        if self.VAO is None:
            self.createVAO()
        
        # ============ VAO for triangle mesh (or point cloud)
        GL.glBindVertexArray( self.VAO[self.MESH_VAO] )
        
        # ======== Vertices
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexBuffers[self.VERTEX_VBO_INDEX])
        # Get the position of the 'position' in parameter of our shader and
        # bind it.
        position = shader.getAttributeLocation("position")
        GL.glEnableVertexAttribArray(position)
        
        # Describe the position data layout in the buffer
        vs = self.vertices.shape[1] # number of columns should be 3 or 4
        GL.glVertexAttribPointer(position, vs, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        
        # Send the data over to the buffer
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices.size*4, self.vertices, self.glDrawMethod)
        
        # ======= Colors
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexBuffers[self.COLOR_VBO_INDEX])
        # Get the position of the 'color' in parameter of our shader and
        # bind it.
        colorPos = shader.getAttributeLocation("color")
        GL.glEnableVertexAttribArray(colorPos)
         
        # Describe the position data layout in the buffer
        cs = self.colors.shape[1] # color format should be 3 (i.e. RGB) or 4 (i.e. RGBA)
        GL.glVertexAttribPointer(colorPos, cs, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
         
        # Send the data over to the buffer
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors.size*4, self.colors, self.glDrawMethod)
        

        # ======= Triangles
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.vertexBuffers[self.TRIANGLE_VBO_INDEX])         
        if self.triangles is not None: 
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.triangles.size*4, self.triangles, self.glDrawMethod)
        else:
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, len(self.vertices)*4, np.arange(len(self.vertices),dtype=np.uint32), self.glDrawMethod)

        # Unbind the VAO first (Important)
        GL.glBindVertexArray( 0 )
        # Unbind other stuff
        GL.glDisableVertexAttribArray(position)
        GL.glDisableVertexAttribArray(colorPos)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        
        
        # ============ VAO for Lines
        if self.triangles is not None: 
            GL.glBindVertexArray( self.VAO[self.LINE_VAO] )
            
            # ======== Vertices
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexBuffers[self.VERTEX_VBO_INDEX])
            # Get the position of the 'position' in parameter of our shader and
            # bind it.
            GL.glEnableVertexAttribArray(position)
            # Describe the position data layout in the buffer
            vs = self.vertices.shape[1] # number of columns should be 3 or 4
            GL.glVertexAttribPointer(position, vs, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
                    
            # ======= Colors
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexBuffers[self.COLOR_VBO_INDEX])
            # Get the position of the 'color' in parameter of our shader and
            # bind it.
            GL.glEnableVertexAttribArray(colorPos) 
            # Describe the position data layout in the buffer
            cs = self.colors.shape[1] # color format should be 3 (i.e. RGB) or 4 (i.e. RGBA)
            GL.glVertexAttribPointer(colorPos, cs, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
             
            # ======= Lines
            # get the edges of the triangles:
            # for a triangle (a,b,c) ==> (a,b, b,c, c,a) edges
            lines = self.triangles.reshape(-1,3)[:,[0,1,1,2,2,0]].reshape(-1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.vertexBuffers[self.LINE_VBO_INDEX])         
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, lines.size*4, lines, self.glDrawMethod)
    
    
            # Unbind the VAO first (Important)
            GL.glBindVertexArray( 0 )
            # Unbind other stuff
            GL.glDisableVertexAttribArray(position)
            GL.glDisableVertexAttribArray(colorPos)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        

        if self.normals is not None: 
            # ============ VAO for Normals 
            GL.glBindVertexArray( self.VAO[self.NORMAL_VAO] )
              
            # ======== Vertices
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexBuffers[self.VERTEX_VBO_INDEX])
            # Get the position of the 'position' in parameter of our shader and
            # bind it.
            GL.glEnableVertexAttribArray(position)
            # Describe the position data layout in the buffer
            vs = self.vertices.shape[1] # number of columns should be 3 or 4
            GL.glVertexAttribPointer(position, vs, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
             
            # ======= Colors
            # Maybe we want to color our normal vectors, so bind this VBO as well. 
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexBuffers[self.COLOR_VBO_INDEX])
            # Get the position of the 'color' in parameter of our shader and
            # bind it.
            GL.glEnableVertexAttribArray(colorPos) 
            # Describe the position data layout in the buffer
            cs = self.colors.shape[1] # color format should be 3 (i.e. RGB) or 4 (i.e. RGBA)
            GL.glVertexAttribPointer(colorPos, cs, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
             
            # ======= Normals
            # create lines from normals
            # The vertices contains all vertices and normal-vertices concatenated (see update method)
            # Each normal line segment is (i,i+len(self.vertices)/2)
            lines = np.arange(len(self.vertices),dtype=np.uint32).reshape(-1,2,order='F').reshape(-1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.vertexBuffers[self.NORMAL_VBO_INDEX])         
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, lines.size*4, lines, self.glDrawMethod)
      
            # Unbind the VAO first (Important)
            GL.glBindVertexArray( 0 )
            # Unbind other stuff
            GL.glDisableVertexAttribArray(position)
            GL.glDisableVertexAttribArray(colorPos)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
         
        self.needsVAOUpdate=False

    def draw(self,shader,flatModelM,showMesh,showFaces,showNormals):
        '''
        Draw this Geometry on the current OpenGL setup.
        '''
        if self.needsVAOUpdate==True:
            return
        try:
            shader.uniform_matrixf("modelM", flatModelM)
            
            if showMesh:
                GL.glBindVertexArray(self.VAO[self.MESH_VAO])
                if self.triangles is not None:
                    GL.glDrawElements(GL.GL_TRIANGLES, self.triangles.size, GL.GL_UNSIGNED_INT , None)
                else:
                    GL.glDrawElements(GL.GL_POINTS, self.vertexCount, GL.GL_UNSIGNED_INT , None)
             
            if showFaces and self.triangles is not None: 
                GL.glBindVertexArray(self.VAO[self.LINE_VAO])
                shader.uniformf("singleColor",0,0,0,1)
                GL.glDrawElements(GL.GL_LINES, self.triangles.size*3, GL.GL_UNSIGNED_INT , None)
                
            if showNormals and self.normals is not None:
                GL.glBindVertexArray(self.VAO[self.NORMAL_VAO])
                shader.uniformf("singleColor",1,0,0,1)
                GL.glDrawElements(GL.GL_LINES, self.vertexCount*2, GL.GL_UNSIGNED_INT , None)

#             TODO: draw normals, etc here.
        finally:
            GL.glBindVertexArray(0)
    
    def cleanUp(self):
        ''' 
        Releases any OpenGL resources held by this Geometry
        Note: cleanUp and __del__ must be called from the thread that owns the GLUT context.
        Calling from another thread will not release the resources and raise a NoContext exception.
        '''
        if self.VAO is not None:
            if self.vertexBuffers is not None:
                GL.glDeleteBuffers(len(self.vertexBuffers),self.vertexBuffers)
                self.vertexBuffers=None
            GL.glDeleteBuffers(len(self.VAO),self.VAO)
            self.VAO=None

class PyGLerModel(object):
    '''
    A Model that can be rendered by PyGLer.
    It uses a Geometry instance which contains vertices, triangles, normals, colors, texture.
    
    The model handles the transformations of the geometry on the model's coordinate space.
     
    '''
    def __init__(self, name, geometry=None, modelM=None):
        self.name = name
        self.geometry=geometry
        self.visible_=True

        self.setModelM(modelM)
        
    
    @property
    def visible(self):
        return self.visible_ 
    
    @visible.setter
    def visible(self,v):
        self.visible_ = v
    
    
    def cleanUp(self):
        self.geometry.cleanUp()
        self._modelM = None        

    def setModelM(self, modelM):
        if modelM is None: # auto center
            vmin,vmax = self.geometry.vertices.min(0), self.geometry.vertices.max(0)
            center = vmin + (vmax-vmin)/2
            
            modelM = np.eye(4,dtype=np.float32)
            modelM[0:3,3] = -center[0:3]
            self._modelM = modelM.transpose().reshape(-1).tolist()

        elif modelM.shape==(4,4):
            self._modelM = modelM.transpose().reshape(-1).tolist()
        else: 
            raise StandardError("Invalid model matrix")
        
    def draw(self,shader,showMesh,showFaces,showNormals):
        if self.visible_:
            self.geometry.draw(shader,self._modelM,showMesh,showFaces,showNormals)   

    def __eq__(self,other):
        return isinstance(other, PyGLerModel) and self.name==other.name        

    @staticmethod
    def LoadObj(filename, computeNormals=False, autoScale=False):
        '''
        Load vertices and faces from a wavefront .obj file and generate normals.
        '''
        data = np.genfromtxt(filename, dtype=[('type', np.character, 1),
                                              ('points', np.float32, 3)])

        # Get vertices and faces
        vertices = data['points'][data['type'] == 'v']
        faces = (data['points'][data['type'] == 'f']-1).astype(np.uint32)

        normals = None
        if computeNormals:
            from utils import ComputeNormals
            normals = ComputeNormals(vertices,faces)
    
        geometry = Geometry(vertices, triangles=faces, normals=normals,autoScale=autoScale)
        return PyGLerModel(filename, geometry)
