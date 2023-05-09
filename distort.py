""" Module for building an image and writing png files,
    written using only the Python standard library.

    Example usage:
        image = Image(50, 50)
        image.setPixel(0, 49, Color(255, 0, 0, 255))
        image.saveAsPNG("redDot.png")
"""

import zlib, struct
import subprocess
import numpy as np
from constants import CN
from PIL import Image as PILImage

class Image(object):
    def __init__ (self,width, height,img):
        self.width, self.height = self.size = width, height
        self.bufferimage = img

    def distortion_correction (self,all_lens):
        distort = lambda x : x + CN[0]*x**3 + CN[1]*x**5
        def check_sign(anew, a):
            if np.sign(a) != np.sign(anew): return abs(anew)*np.sign(a)
            return anew
        width, height = bigger = ((self.width*1,self.height*1))
        self.distortedimage = PILImage.new("RGBA",bigger)
        big = self.bufferimage.resize(bigger)

        for a in range(width):
            for b in range(height):
                for lens in all_lens:
                    x = (2*a-width)/width
                    y = (2*b-height)/height
                    # converted to [-1,1]

                    coords = np.array((x,y))
                    r = np.linalg.norm(coords-np.array(lens))
                    theta = np.arctan(y/x) if x else 0

                    rnew = distort(r)
                    xnew = float(rnew*np.cos(theta))
                    ynew = float(rnew*np.sin(theta))

                    xnew = check_sign(xnew, x)
                    ynew = check_sign(ynew,y)
                    
                    xnew = int(((xnew+1)*width)//2)
                    ynew = int(((ynew+1)*height)//2)
                    cnew = (xnew, ynew)

                    pxl = big.getpixel((a,b))
                    try:
                        self.distortedimage.putpixel(cnew,pxl)
                    except: pass

        self.distortedimage = self.distortedimage.resize(self.size)

        return self.distortedimage


if __name__ == "__main__":
    for i in range(6800//16):
        img = Image(300,300,PILImage.open(f'./frames/{i*16}.png'))
        img.distortion_correction([0,0]).save(f'./distort/{i}.png')
        print(float(i)/6800)
