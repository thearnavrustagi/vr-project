""" Module for reading a .obj file into a stored model,
    retrieving vertices, faces, properties of that model.
    Written using only the Python standard library.
"""

from vector import Vector, vec_mat_mul
from math import pi
import numpy as np
import transform
from constants import DRAG_COEFFECIENT, AIR_DENSITY, AREA


class Model(object):
    def __init__(self, file, gravity=False, mass = 1,momentum=Vector(0,0,0),scale=Vector(1,1,1),displacement=Vector(0,0,0)):
        self.vertices = []
        self.faces = []
        self.mass = mass

        self.displacement = displacement
        self.momentum = momentum
        self.force = Vector(0, 0, 0)

        self.angular_velocity = Vector(0,0,0)
        self.rotation = Vector(0,0,0)
        self.net_rotation = Vector(0,0,0)

        self.scale = scale


        self.filename = file
        self.collided = False

        self.collision_radius = 0.4

        if gravity:
            self.force += Vector(0, -0.5, 0)


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

        self.displacement += (self.momentum/self.mass)*delta + (self.force/self.mass)*delta**2
        self.momentum += self.force*delta
        self.rotation += self.angular_velocity * delta
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

    def collide_with (self, model, restitution=1):
        if self.collided: return
        print('collision!')
        moi = lambda M,r : 0.4*M*r**2
        model.calculate_centroid(), self.calculate_centroid()
        v_a, v_b = self.momentum/self.mass, model.momentum/model.mass
        m_a,m_b = self.mass, model.mass

        I_a,I_b = moi(m_a,self.collision_radius), moi(m_b,model.collision_radius)
        I_a = np.array([[I_a,0.0,0.0],[0.0,I_a,0.0],[0.0,0.0,I_a]])
        I_b = np.array([[I_b,0,0],[0,I_b,0],[0,0,I_b]])

        ratio = (1.0*self.collision_radius)/model.collision_radius
        section = lambda a,b,m : (a/m + m*b)/(m+1/m)
        
        x = section(self.centroid.x,model.centroid.x,ratio)
        y = section(self.centroid.y,model.centroid.y,ratio)
        z = section(self.centroid.z,model.centroid.z,ratio)
        coll_pt = Vector(x,y,z)

        r_a, r_b = self.centroid-coll_pt, model.centroid - coll_pt

        normal = (self.centroid - model.centroid).normalize()

        vai, wai = self.momentum/self.mass, self.angular_velocity
        vbi, wbi = model.momentum/model.mass, model.angular_velocity

        vaf,vbf, waf, wbf = self.calculate_collision (restitution,m_a,m_b,I_a,I_b,r_a,r_b,normal,vai,vbi,wai,wbi)

        self.momentum, self.angular_velocity = vaf*self.mass, waf
        model.momenutm, model.angular_velocity = vbf*model.mass, wbf

        self.collided = True
        
    def calculate_collision(self,e,ma,mb,Ia,Ib,ra,rb,n,vai,vbi,wai,wbi):
        IaInverse = np.linalg.inv(Ia)
        normal = n.normalize()
        angularVelChangea = normal.copy()
        angularVelChangea.cross(ra)
        angularVelChangea = vec_mat_mul(IaInverse,angularVelChangea)
        vaLinDueToR = angularVelChangea.copy().cross(ra)
        scalar = 1/ma + vaLinDueToR.dot(normal)
        IbInverse = np.linalg.inv(Ib)
        angularVelChangeb = normal.copy()
        angularVelChangeb.cross(rb)
        angularVelChangeb = vec_mat_mul(IbInverse,angularVelChangeb)
        vbLinDueToR = angularVelChangeb.copy().cross(rb)
        scalar += 1/mb + vbLinDueToR.dot(normal)
        Jmod = (e+1)*(vai-vbi).norm()/scalar
        J = normal * (Jmod)
        vaf = vai - J*(1/ma)
        vbf = vbi - J*(1/mb)
        waf = wai - angularVelChangea
        wbf = wbi - angularVelChangeb
        return vaf, vbf, waf, wbf



    def apply_transform (self,v):
        d = self.net_displacement.components
        r = self.rotation.components
        s = self.scale.components
        return transform.all(v, d, r, s)
