from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector
import transform
import pandas as pd

from constants import FOV
from time import time, sleep
from qmath import euler

import numpy as np

width = 300
height = 300
img_color =  Color(171, 209, 255, 255) 
image = Image(width, height,img_color)
# in polar coordinates (theta, len)
lens_locations = [(-0,0)]

# Init z-buffer
zBuffer = [-float('inf')] * width * height

# Load the model
models = [Model('data/headset.obj',displacement=Vector(0,0,0))]#, Model('data/headset.obj', gravity=True, momentum=Vector(0,0,0),scale=Vector(0.2,0.2,0.2),displacement=Vector(0,2,0))]
[model.normalizeGeometry() for model in models]

def main():
    global models
    prev_time = 0
    gyroscope = pd.read_csv('./dataset/gyroscope.csv')
    time_df = pd.read_csv ('./dataset/time.csv')
    for i, row in gyroscope.iterrows():
        delta = time_df.iloc[[i]].to_numpy()[0][1] - prev_time
        print("done",(float(i)/6958))
        renderable, models = physics_process(models,delta,row)
        prev_time += delta
        render_scene(renderable)


def physics_process (models, delta, row):
    renderable = []
    models = check_collisions(models)
    for model in models:
        model.angular_velocity = Vector(*tuple(row[1:]))
        transformed = model.apply_physics(delta)
        transformed.transform()
        transformed.normalizeGeometry()

        renderable.append(transformed)


    return renderable, models

def check_collisions(models):
    for i,model in enumerate(models):
        for other in models[i+1:]:
            thresh = other.collision_radius + model.collision_radius
            if (other.displacement-model.displacement).norm() < thresh:
                other.collide_with(model)
    return models

def render_scene(models):
    global image, zBuffer

    zBuffer = [-float('inf')] * width * height
    image = Image(width, height, img_color)

    for model in models:
        render_model(model)

    image.close()

def getOrthographicProjection(x, y, z):
    # Convert vertex from world space to screen space
    # by dropping the z-coordinate (Orthographic projection)
    screenX = int((x+1.0)*width/2.0)
    screenY = int((y+1.0)*height/2.0)

    return screenX, screenY

# fov tuple containing x and y fov
'''
def getPerspectiveProjection(x, y, z, fov:tuple):
       # uses simple trignometry to get the location on the screen
    capture_size = tuple(map(lambda a,b,c : c+2*a*np.tan(b) ,(z,z),fov,(width,height)))
    delta = tuple(map(lambda a,b : a*np.tan(b) ,(z,z),fov))
    scale = (capture_size[0]/width, capture_size[1]/height)
#   d = 5
#   screenX = int(x*d/z)*5
#   screenY = int(y*d/z)*5#height
    screenX, screenY = getOrthographicProjection(x,y,z)
    return int(screenX*scale[0]+delta[0]), int(screenY*scale[1]+delta[1])

'''
def getVertexNormal(vertIndex, faceNormalsByVertex):
    # Compute vertex normals by averaging the normals of adjacent faces
    normal = Vector(0, 0, 0)
    for adjNormal in faceNormalsByVertex[vertIndex]:
        normal = normal + adjNormal

    return normal / len(faceNormalsByVertex[vertIndex])

def getPerspectiveProjection (x,y,z,n,f):
#    x,y = getOrthographicProjection(x,y,z)
    matrix = np.array([
        [n,0,0,0],
        [0,n,0,0],
        [0,0,(f+n),-f*n],
        [0,0,1,0]])
    
    res = np.matmul(np.array([x,y,z,1]),matrix)

    return int(res[0]/res[3]), int(res[1]/res[3])



def render_model(model):
    global zBuffer, width, height, FOV, image, img_color

    # Calculate face normals
    faceNormals = {}
    for face in model.faces:
        p0, p1, p2 = [model.vertices[i] for i in face]
        faceNormal = (p2-p0).cross(p1-p0).normalize()

        for i in face:
            if not i in faceNormals:
                faceNormals[i] = []

            faceNormals[i].append(faceNormal)

    # Calculate vertex normals
    vertexNormals = []
    for vertIndex in range(len(model.vertices)):
        vertNorm = getVertexNormal(vertIndex, faceNormals)
        vertexNormals.append(vertNorm)

    #projectionFunction = lambda x,y,z : getPerspectiveProjection(x,y,z,0.1,10)
    projectionFunction = getOrthographicProjection


    # Render the image iterating through faces
    for face in model.faces:
        p0, p1, p2 = [model.vertices[i] for i in face]
        n0, n1, n2 = [vertexNormals[i] for i in face]

        # Define the light direction
        lightDir = Vector(0, 0, -1)

        # Set to true if face should be culled
        cull = False

        # Transform vertices and calculate lighting intensity per vertex
        transformedPoints = []
        for p, n in zip([p0, p1, p2], [n0, n1, n2]):
            intensity = n * lightDir

            # Intensity < 0 means light is shining through the back of the face
            # In this case, don't draw the face at all ("back-face culling")
            if intensity < 0:
                cull = True
                break
            color = Color(intensity*255, intensity*255, intensity*255, 255)

            vec = (p.x, p.y, p.z)
            #vec = model.apply_transform(vec)
            screenX, screenY = projectionFunction(vec[0],vec[1],vec[2])
            transformedPoints.append(Point(screenX, screenY, p.y, color))

        if not cull:
            Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw(image, zBuffer)

    #image.distort(lens_locations)
    image.saveAsPNG("image.png")

if __name__ == "__main__":
    main()
