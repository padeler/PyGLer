'''
Created on Oct 7, 2014

Utility module.

Use the CameraParams class to set the intrinsics for the viewport camera of PyGLer. 

@author: padeler
'''

import numpy as np

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
    tris = vertices[faces]
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
        
    return r;


class CameraParams(object):
    '''
    Virtual Camera used to for the PyGLer Viewport.
    '''
    
    def __init__(self,width=640,height=480,cx=320,cy=240,fx=575.81573,fy=575.81573,znear=1.0,zfar=10000.0,unit=1.0):
        '''
        Camera Params constructor. The default values correspond to the default 
        Kinect camera instrinsics (provided by the OpenNI driver)
        '''
        
        self._width = width
        self._height = height
        self._cx = cx
        self._cy = cy
        self._fx = fx
        self._fy = fy
        
        self._unit=unit
        self._zfar=zfar
        self._znear=znear
            
        #Create a (row major) projection matrix from intrinsics. 
        intr = np.zeros((4,4),dtype=np.float32); 
        intr[0][0] = (2.0 * fx) / width;
        intr[0][1] = 0;
        intr[0][2] = -1 + (2 * cx) / width;
        intr[1][1] = (-2 * fy) / height
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

    @property
    def projectionMat(self):
        return self._projectionMat
    
    
    
    