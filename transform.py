import numpy as np

def preprocess (vectors : tuple) -> tuple :
    ans = []
    for vector in vectors:
        ans.append(np.array(vector))

    return tuple(ans)

def transform (vector, transform_matrix):
    return np.matmul(vector,transform_matrix)[:3]

def translate (vector, translate) -> np.ndarray :
    vector, translate = preprocess((vector, translate))

    return [v+t for v,t in zip(vector,translate)]

def rotate_x (vector, theta) -> np.ndarray :
    vector = preprocess((vector,))
    vector = np.append(vector,1)

    transform_matrix = np.array([
        [1,0,0,0],
        [0,np.cos(theta),-np.sin(theta),0],
        [0,np.sin(theta),np.cos(theta),0],
        [0,0,0,1]])

    return transform(vector, transform_matrix)


def rotate_y (vector, theta) -> np.ndarray :
    vector = preprocess((vector,))
    vector = np.append(vector,1)

    transform_matrix = np.array([
        [np.cos(theta),0,np.sin(theta),0],
        [0,1,0,0],
        [-np.sin(theta),0,np.cos(theta),0],
        [0,0,0,1]])

    return transform(vector, transform_matrix)


def rotate_z (vector, theta) -> np.ndarray :
    vector = preprocess((vector,))
    vector = np.append(vector,1)

    transform_matrix = np.array([
        [np.cos(theta),-np.sin(theta),0,0],
        [np.sin(theta),np.cos(theta),0,0],
        [0,0,1,0],
        [0,0,0,1]])

    return transform(vector, transform_matrix)

def rotate (vector, theta) -> np.ndarray :
    rot_func = [rotate_x, rotate_y, rotate_z]
    for rfnc, thet in zip(rot_func,theta):
        vector = rfnc(vector,thet)

    return vector

def scale (vector, factor) -> np.ndarray :
    vector = preprocess((vector,))[0]
    print(vector,factor)

    a = np.array([dim*s for dim, s in zip(vector, factor)])
    print(a)

    return a

def all (point, displacement, rotation, scale_factor):
    fncs = [translate, rotate, scale]
    data = [displacement, rotation, scale_factor]

    for fn, vec in zip(fncs, data):
        point = fn(point,vec)

    return tuple(point)
