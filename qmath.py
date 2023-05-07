import numpy as np
import math

# takes radians, returns quaternion
def quaternion(theta,v=(0,0,1)):
    x = v[0]*np.sin(theta/2)
    y = v[1]*np.sin(theta/2)
    z = v[2]*np.sin(theta/2)

    w = np.cos(theta/2)
    
    return (x, y, z, w)

# returns identity vector
def unit (v):
    n = np.linalg.norm(v)
    return (np.array(v)/n)

# returns radians
def euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(t3, t4))

    return tuple(np.deg2rad((roll, pitch, yaw)))

def qconjugate(x,y,z,w):
    return -x,-y,-z,w

def qmagnitude (x,y,z,w):
    return np.linalg.norm([x,y,z,w])

def qinverse (q):
    return qconjugate(q) / (quaternion_multiply(q,qconjugate(q)))

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = tuple(quaternion0)
    w1, x1, y1, z1 = tuple(quaternion1)
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def euler2quat(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return (qx, qy, qz, qw)

def euler_between(vec1, vec2):
    ans =  np.arcsin(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return ans


if __name__ == "__main__":
    print(quaternion_multiply((1,2,3,4),()))
