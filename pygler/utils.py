'''
Created on Oct 7, 2014

Utility module.

Use the CameraParams class to set the intrinsics for the viewport camera of PyGLer. 

@author: padeler
'''

import numpy as np
from pygler.model import PyGLerModel,Geometry

def normalize_Nx3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def ComputeNormals(vertices, faces):
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros( vertices.shape, dtype=vertices.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = np.copy(vertices[faces])
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )# n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    k = normalize_Nx3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[ faces[:,0] ] += k
    norm[ faces[:,1] ] += k
    norm[ faces[:,2] ] += k
    r = normalize_Nx3(norm)
        
    return r


def CreateAxisModel(name="axis",thickness=0.01,length=1.0,colorX=[255,0,0],colorY=[0,255,0],colorZ=[0,0,255],alpha=1.0):
    '''
    Create a PyGLerModel instance that represents a 3D axis. 
    Usefull for visualizing camera positions and starting positions of models
    '''
    
    s = thickness/2.0
    lr = length / s
    
    cubeV = np.array([-s, s, s,
                       s, s, s,
                       s,-s, s,
                      -s,-s, s,
                      -s,-s,-s,
                      -s, s,-s,
                       s, s,-s,
                       s,-s,-s],dtype='f').reshape(-1,3)
    
    cubeF = np.array([0,1,2, 2,3,0, # front , lengthen [z coord] for Z+ axis 
                      2,3,4, 4,7,2, # bottom
                      0,5,4, 4,3,0, # left
                      7,2,1, 1,6,7, # right , lengthen [x coord] for X+axis 
                      5,6,7, 7,4,5, # back
                      0,5,6, 6,1,0  # top , lenghten [y coord] for Y+ axis
                                    ],dtype=np.uint32)

    
    
    axisV = np.empty((24,3),dtype=np.float32)
    axisF = np.empty((36*3),dtype=np.uint32) # each axis has 6 faces with 2 triangles each face, so 36 points for each axis
    
    Xaxis = np.copy(cubeV)
    Xaxis[[1,2,6,7],0] *= lr

    Yaxis = np.copy(cubeV)
    Yaxis[[0,1,5,6],1] *= lr
    
    Zaxis = np.copy(cubeV)
    Zaxis[[0,1,2,3],2] *= lr
    
    for i,ax in enumerate((Xaxis,Yaxis,Zaxis)):
        axisV[i*8:(i+1)*8] = ax
        axisF[(i*36):((i+1)*36)] = np.copy(cubeF) + i*8 
    
    # colors.
    colors = np.empty((24,3),dtype=np.ubyte)
    colors[0*8:1*8] = colorX
    colors[1*8:2*8] = colorY
    colors[2*8:3*8] = colorZ
    
    return PyGLerModel(name, geometry=Geometry(axisV, axisF, colors=colors,alpha=alpha,autoScale=False,bgrColor=False),modelM=np.eye(4,dtype=np.float32))
    

def CreateCubeModel(name="cube",side=1.0,scale=[1.0,1.0,1.0],colors=[0,255,0],alpha=0.9):
    '''
    Create a PyGLerModel instance that represents a unit cube scaled by a given value along each axis. 
    Usefull for visualizing positions of models relative to each other.
    '''
    
    s = side/2.0
    cubeV = np.array([-s, s, s,
                       s, s, s,
                       s,-s, s,
                      -s,-s, s,
                      -s,-s,-s,
                      -s, s,-s,
                       s, s,-s,
                       s,-s,-s],dtype='f').reshape(-1,3)
    
    cubeF = np.array([0,1,2, 2,3,0, # front , lengthen [z coord] for Z+ axis 
                      2,3,4, 4,7,2, # bottom
                      0,5,4, 4,3,0, # left
                      7,2,1, 1,6,7, # right , lengthen [x coord] for X+axis 
                      5,6,7, 7,4,5, # back
                      0,5,6, 6,1,0  # top , lenghten [y coord] for Y+ axis
                                    ],dtype=np.uint32)

    
    cubeV *= scale
        
    colors = np.array(colors,dtype=np.ubyte)
    
    return PyGLerModel(name, geometry=Geometry(cubeV, cubeF, colors=colors,autoScale=False,alpha=alpha),modelM=np.eye(4,dtype=np.float32))


class CameraParams(object):
    '''
    Virtual Camera used to for the PyGLer Viewport.
    '''
    
    def __init__(self,width=640,height=480,
                    cx=320,cy=240,fx=575.81573,fy=575.81573,
                    znear=1.0,zfar=10000.0,unit=1.0, 
                    position=[0, 0, 0], rotation=[0, 0, 0]):
        '''
        Camera Params constructor. The default values correspond to the default 
        Kinect camera instrinsics (provided by the OpenNI driver)
        
        legacyMode created a projection matrix that is compatible with the standard MBV projection matrix.
         
        '''
        self.width = width
        self.height = height
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        
        self._unit=unit
        self._zfar=zfar
        self._znear=znear
            
        #Create a (row major) projection matrix from intrinsics. 
        intr = np.zeros((4,4),dtype=np.float32); 
        intr[0][0] = (2.0 * fx) / width;
        intr[0][1] = 0;
        intr[0][2] = -1 + (2 * cx) / width;
        
        intr[1][1] = -(2 * fy) / height # FIXME This "-" is compatible with mbv but not standard. 
            
        intr[1][2] = 1 - (2 * cy) / height
        intr[2][2] = 1;
        intr[3][3] = unit; # unit conversion -- If the extrinsics are in meters set unit=1000 to convert all to meters
        cpm = np.zeros((4,4),dtype=np.float32) # clipped projection to znear through zfar
        cpm[0][0] = 1;
        cpm[1][1] = 1;
        cpm[2][2] = zfar/(zfar - znear);
        cpm[2][3] = (-((zfar*znear)/(zfar - znear))) / unit;
        cpm[3][2] = 1;
        

            
        projectionMat = cpm.dot(intr)
        
        self._projectionMat = projectionMat.transpose()

    @staticmethod
    def from_file(calib_file):
        import yaml
        with open(calib_file) as f:
            calib = yaml.safe_load(f)
        return CameraParams(**calib)
    
    @property
    def projectionMat(self):
        return self._projectionMat
    
    