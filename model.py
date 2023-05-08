""" Module for reading a .obj file into a stored model,
    retrieving vertices, faces, properties of that model.
    Written using only the Python standard library.
"""

from vector import Vector
from math import pi
import transform
from constants import DRAG_COEFFECIENT, AIR_DENSITY, AREA


class Model(object):
    def __init__(self, file, gravity=False, mass = 1):
        self.vertices = []
        self.faces = []
        self.mass = mass

        self.displacement = Vector(0, 0, 0)
        self.net_displacement = Vector(0, 0, 0)
        self.momentum = Vector(0, 0, 0)
        self.force = Vector(0, 0, 0)

        self.angular_momentum = Vector(0,0,0)
        self.rotation = Vector(0,0,0)
        self.net_rotation = Vector(0,0,0)

        self.scale = Vector(1,1,1)


        self.filename = file

        if gravity:
            self.force += Vector(0, 0, 0)


        # Read in the file
        f = open(file, 'r')
        for line in f:
            if line.startswith('#'):
                continue
            segments = line.split()
            if not segments:
                continue

            # Vertices
            if segments[0] == 'v':
                vertex = Vector(*[float(i) for i in segments[1:4]])
                self.vertices.append(vertex)

            # Faces
            elif segments[0] == 'f':
                # Support models that have faces with more than 3 points
                # Parse the face as a triangle fan
                for i in range(2, len(segments)-1):
                    corner1 = int(segments[1].split('/')[0])-1
                    corner2 = int(segments[i].split('/')[0])-1
                    corner3 = int(segments[i+1].split('/')[0])-1
                    self.faces.append([corner1, corner2, corner3])
        
    def normalizeGeometry(self):
        maxCoords = [0, 0, 0]

        for vertex in self.vertices:
            maxCoords[0] = max(abs(vertex.x), maxCoords[0])
            maxCoords[1] = max(abs(vertex.y), maxCoords[1])
            maxCoords[2] = max(abs(vertex.z), maxCoords[2])

        for vertex in self.vertices:
            vertex.x = vertex.x / maxCoords[0]
            vertex.y = vertex.y / maxCoords[1]
            vertex.z = vertex.z / maxCoords[2]

    def apply_physics (self, delta):
        force = self.compute_force()

        self.momentum += self.force*delta
        self.displacement += (self.momentum/self.mass)*delta
        self.rotation += self.angular_momentum * delta
        self.rotation = Vector(*tuple(c%(2*pi) for c in self.rotation.components))

        model = Model(self.filename, gravity = True)
        model.displacement = self.displacement
        model.rotation = self.rotation 
        model.scale = self.scale

        return model

    
    def calculate_centroid(self):
        self.centroid = Vector(0,0,0)

        for v in self.vertices: 
            self.centroid += v

        self.centroid /= len(self.vertices)

    def compute_force (self):
        drag = DRAG_COEFFECIENT*AIR_DENSITY*AREA/2
        vec = Vector(*tuple([(a*b)/self.mass**2 for a,b in zip(self.momentum.components,self.momentum.components)]))
        return self.force - vec

    def transform (self):
        self.calculate_centroid()
        d = self.displacement.components
        r = self.rotation.components
        s = self.scale.components

        for i, v in enumerate(self.vertices):
            vec = (self.vertices[i]-self.centroid).components
            vec = transform.rotate(vec,r)
            vec = transform.scale(vec,s)
            vec = transform.translate(vec,d)
            vec = tuple(i+j for i,j in zip(vec,self.centroid.components))
            self.vertices[i] = Vector(*vec)

    def apply_transform (self,v):
        d = self.net_displacement.components
        r = self.rotation.components
        s = self.scale.components
        return transform.all(v, d, r, s)
