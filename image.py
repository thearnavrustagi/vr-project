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

class Color(object):
    """ A small class representing a 32-bit RGBA color."""
    def __init__(self, r, g, b, a):
        self.color = (r, g, b, a)

    def r(self):
        return self.color[0]

    def g(self):
        return self.color[1]

    def b(self):
        return self.color[2]

    def a(self):
        return self.color[3]

    def getTuple(self):
        return self.color

    def getHexString(self):
        return "0x%02X%02X%02X%02X" % self.color

    def getByteString(self):
        """ Pack the color as a C-style byte string."""
        return struct.pack('>4B', self.color[0], self.color[1], self.color[2], self.color[3])

    def getAlphaBlend(self, destColor):
        """ Alpha blend this color with the provided destination color."""
        alpha = self.a() / 255
        outR = int(self.r() * alpha) + int(destColor.r() * (1 - alpha))
        outG = int(self.g() * alpha) + int(destColor.g() * (1 - alpha))
        outB = int(self.b() * alpha) + int(destColor.b() * (1 - alpha))
        outA = self.a() + int(destColor.a() * (1-alpha))
        return Color(outR, outG, outB, outA)

class Image(object):
    """ An image class capable of generating and saving a PNG.
        Attributes:
            width: The width of the image
            height: The height of the image
            buffer: Representation of the image storing Color values for each pixel
    """
    def __init__(self, width, height, color = Color(0, 0, 0, 255)):
        """ Create the buffer, fill it with black pixels."""
        self.width = width
        self.height = height
        self.size = (width, height)

        # Each row consists of a null byte followed by colors for each pixel
        row = bytearray(1) + bytearray([color.r(), color.g(), color.b(), color.a()] * width)
        self.buffer = row * height

        self.bufferimage = None

    def setPixel(self, x, y, color):
        """ Set the color value for the pixel at (x, y)."""
        if (x not in range(0, self.width)) or (y not in range (0, self.height)):
            return

        # Flip Y coordinate so that up is positive
        flipY = (self.height - y - 1)
        index = (flipY * self.width + x) * 4 + flipY

        # Get the existing destination color
        destColor = Color(*tuple(self.buffer[index + 1 : index + 5]))

        # Blend the new color with the destination color
        outColor = color.getAlphaBlend(destColor)

        # Set the new pixel colors in the buffer
        self.buffer[index + 1 : index + 5] = outColor.getTuple()

    def saveAsPNG(self, filename = "render.png"):
        """ Pack a new buffer formatted as a PNG, then save it to a file."""
        print("Saving PNG...")

        def makeChunk(chunkType, chunkData):
            """ Pack data into standard PNG chunks. Chunks consist of:
                    - a 4-byte length 
                    - a 4-byte chunk type 
                    - the chunk data (compressed)
                    - a 4-byte cyclic redundancy check value (CRC)
            """
            chunk = struct.pack(">I", len(chunkData)) + \
                    chunkType + \
                    chunkData +	\
                    struct.pack(">I", 0xFFFFFFFF & zlib.crc32(chunkType + chunkData))

            return chunk

        # Compose the PNG out of 3 chunks using the above function:
        #	- IHDR: Header containing image size, color depth, existence of alpha channel etc. (See PNG spec)
        #	- IDAT: Chunk containing the actual image data
        #	- IEND: End-of-image chunk

        # Start with the universal PNG signature identifying the file as a PNG, then append chunks
        packedData = b'\x89PNG\r\n\x1a\n' +	\
                     makeChunk(b'IHDR', struct.pack(">2I5B", self.width, self.height, 8, 6, 0, 0, 0)) + \
                     makeChunk(b'IDAT', zlib.compress(self.buffer, 9)) + \
                     makeChunk(b'IEND', b'')

        png = open(filename, 'wb')
        png.write(packedData)
        png.close()

    def show (self,distortion=False,lens=None):
        self.bufferimage = PILImage.new("RGBA",self.size)
        
        for row_no in range(len(self.buffer[::1+self.width*4])):
            rindex = lambda row_no, width: row_no*width*4+row_no
            r = rindex(row_no,self.width)
            row = self.buffer[r+1:rindex(row_no+1,self.width)]
            for i in range(len(row[::4])):
                index = i*4
                pixel = tuple(row[index:index+4])
                x,y = location = (i,row_no)
                self.bufferimage.putpixel(location, pixel)
        
        if distortion:
            self.bufferimage = self.distortion_correction(lens)
        
        self.bufferimage.save(".temp.png")
        self.viewer = subprocess.Popen(['xdg-open', './.temp.png'])


    def distortion_correction (self,all_lens):
        distort = lambda x : x + CN[0]*x**3 + CN[1]*x**5
        def check_sign(anew, a):
            if np.sign(a) != np.sign(anew): return abs(anew)*np.sign(a)
            return anew
        self.distortedimage = PILImage.new("RGBA",self.size)
        

        for a in range(self.width):
            for b in range(self.height):
                for lens in all_lens:
                    x = (2*a-self.width)/self.width
                    y = (2*b-self.height)/self.height
                    coords = np.array((x,y))
                    r = np.linalg.norm(coords-np.array(lens))
                    theta = np.arctan(y/x) if x else 0
                    rnew = distort(r)
                    xnew = float(r*np.cos(theta))
                    ynew = float(r*np.sin(theta))

                    xnew = check_sign(xnew, x)
                    ynew = check_sign(ynew,y)
                    
                    xnew = int(((xnew+1)*self.width)//2)
                    ynew = int(((ynew+1)*self.height)//2)
                    cnew = (xnew, ynew)

                    pxl = self.bufferimage.getpixel(tuple(coords))
                    try:
                        self.distortedimage.putpixel(cnew,pxl)
                    except: pass

        return self.distortedimage
                    

    def close(self):
        if self.bufferimage:
            self.viewer.terminate()
            self.viewer.kill()  # make sure the viewer is gone; not needed on Windows

