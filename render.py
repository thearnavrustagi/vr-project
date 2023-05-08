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

width = 500
height = 500
img_color =  Color(171, 209, 255, 255) 
image = Image(width, height,img_color)
# in polar coordinates (theta, len)
lens_locations = [(-0,0)]

# Init z-buffer
zBuffer = [-float('inf')] * width * height

# Load the model
models = [Model('data/headset.obj', gravity=True)]
[model.normalizeGeometry() for model in models]

def main():
    prev_time = 0
    gyroscope = pd.read_csv('./dataset/gyroscope.csv')
    time_df = pd.read_csv ('./dataset/time.csv')
    for i, row in gyroscope.iterrows():
        delta = 1#time_df.iloc[[i]].to_numpy()[0][1] - prev_time
        print("done",(float(i)/6958))
        renderable = physics_process(models,delta,row)
        prev_time += delta
        render_scene(renderable)


def physics_process (models, delta, row):
    renderable = []
    for model in models:
        model.angular_momentum = Vector(*tuple(row[1:]))
        transformed = model.apply_physics(delta)
        transformed.transform()
        transformed.normalizeGeometry()

        renderable.append(transformed)
    return renderable

def render_scene(models):
    for model in models:
        render_model(model)

def getOrthographicProjection(x, y, z):
    # Convert vertex from world space to screen space
    # by dropping the z-coordinate (Orthographic projection)
    screenX = int((x+1.0)*width/2.0)
    screenY = int((y+1.0)*height/2.0)

    return screenX, screenY


# fov tuple containing x and y fov
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

    return screenX, screenY 
def getVertexNormal(vertIndex, faceNormalsByVertex):
    # Compute vertex normals by averaging the normals of adjacent faces
    normal = Vector(0, 0, 0)
    for adjNormal in faceNormalsByVertex[vertIndex]:
        normal = normal + adjNormal

    return normal / len(faceNormalsByVertex[vertIndex])



def render_model(model):
    global zBuffer, width, height, FOV, image, img_color

    image = Image(width, height, img_color)
    zBuffer = [-float('inf')] * width * height
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

    projectionFunction = lambda x,y,z : getPerspectiveProjection(x,y,z,FOV)
    #projectionFunction = getOrthographicProjection


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
    image.close()
    image.show()

if __name__ == "__main__":
    main()
