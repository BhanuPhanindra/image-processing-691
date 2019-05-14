#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import scipy as sc
from scipy import signal
from skimage import filters
from matplotlib import pyplot as plt
import cv2
import time
import math
import matplotlib.image as mpimg
# from pillow import Image
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:





# In[14]:





# In[ ]:





# In[166]:


def GaussianBlurImage(image, sigma):
    t1 = time.time()
    filter_size = 2 * int(sigma * 4 + 0.5) + 1
#     print(f'filter size: {filter_size}')
    
    #Creating Filter
    gauss_filter = np.zeros((filter_size, filter_size, 3), dtype=np.float32)
    gauss_filter_2d = np.zeros((filter_size, filter_size), dtype=np.float32)
    for i in range(filter_size):
        for j in range(filter_size):
            x = i - filter_size // 2
            y = j - filter_size // 2
            tmp = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2)/(2 * sigma ** 2))
            gauss_filter[i, j] = np.array([tmp, tmp, tmp])
            gauss_filter_2d[i,j] = tmp
    
    #Reading image using import image
    img = Image.open(image)
    width = img.width
    height = img.height
    pixx = img.load()
    #Remove filter_size -1 if zero padding not required
    pix = np.zeros((width + filter_size -1, height + filter_size -1, 3), dtype=np.float32) #Numpy version of image
    
        
    #Copying pixel values to numpy array
    for i in range(width):
        for j in range(height):
            pix[i + filter_size // 2, j + filter_size // 2] = pixx[i,j]
            
#     #CONVOLUTION WITHOUT ZERO PADDING
#     print('convolution started')
#     for i in range(0 + (filter_size // 2), width -(filter_size // 2)):
#         for j in range(0 + (filter_size // 2), height - (filter_size // 2)):
#             cPos = (i,j)
#             tmp = pix[cPos[0] - (filter_size // 2):cPos[0] + (filter_size // 2)+1, cPos[1] - (filter_size // 2):cPos[1] + (filter_size // 2) +1]
#             tmp = tmp * gauss_filter
#             tmp = tmp.sum(axis=0)
#             tmp = tmp.sum(axis=0)
#             pixx[i,j] = (tmp[0], tmp[1], tmp[2])
            
            
    #CONVOLUTION WITH ZERO PADDING
#     print('convolution started')
    for i in range(0 , width ):
        for j in range(0 , height):
            cPos = (i,j)
            tmp = pix[cPos[0]:cPos[0] + (filter_size), cPos[1]:cPos[1] + (filter_size)]
            tmp = tmp * gauss_filter
            tmp = tmp.sum(axis=0)
            tmp = tmp.sum(axis=0)
            pixx[i,j] = (tmp[0], tmp[1], tmp[2])

#     img.save('Gogh/' + str(sigma) + '.png')
    img.save('1.png')
#     print("Sucessfully created and saved as 1.png")
    print("Time Taken(sec):", time.time()-t1)


# In[ ]:





# In[131]:


# =============================================================================
# for i in range(2,9, 2):
#     GaussianBlurImage('hw1_data/Gogh.png', i)
# =============================================================================


# In[ ]:





# In[ ]:





# In[167]:


# for i in range(2,17,2):
#     print(i)
GaussianBlurImage('hw1_data/Seattle.jpg', 2)


# In[121]:


GaussianBlurImage('6b.png', 1)


# In[168]:


def SeparableGaussianBlurImage (image, sigma):
    t1 = time.time()
    filter_size = 2 * int(sigma * 4 + 0.5) + 1
#     print(f'filter size: {filter_size}')
    
    #Creating Filter
    gauss_filter_x = np.zeros((filter_size, 3), dtype=np.float32)
    gauss_filter_y = np.zeros((filter_size, 3), dtype=np.float32)
    gauss_filter_2d = np.zeros((filter_size, 1), dtype=np.float32)
    
#     for i in range(filter_size):
#         for j in range(filter_size):
#             x = i - filter_size // 2
#             y = j - filter_size // 2
#             tmp_x = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 )/(2 * sigma ** 2))
#             tmp_y = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-(y ** 2)/(2 * sigma ** 2))
#             gauss_filter_x[i, j] = np.array([tmp_x, tmp_x, tmp_x])
#             gauss_filter_y[i, j] = np.array([tmp_y, tmp_y, tmp_y])
#             gauss_filter_2d[i,j] = tmp


    for i in range(filter_size):
        x = i - filter_size // 2
        tmp_x = 1.0 / ((2 * np.pi * sigma ** 2) ** 0.5) * np.exp(-(x ** 2 )/(2 * sigma ** 2))
        gauss_filter_x[i] = np.array([tmp_x, tmp_x, tmp_x])
        gauss_filter_2d[i,0] = tmp_x
    
    for i in range(filter_size):
        y = i - filter_size // 2
        tmp_x = 1.0 / ((2 * np.pi * sigma ** 2) ** 0.5) * np.exp(-(y ** 2 )/(2 * sigma ** 2))
        gauss_filter_y[i] = np.array([tmp_x, tmp_x, tmp_x])
    
#     print(gauss_filter_x.sum(axis=0))
#     print(gauss_filter_2d.sum(axis=0))
#     print(gauss_filter_y.sum(axis=0))



    #Reading image using import image
    img = Image.open(image)
    width = img.width
    height = img.height
    pixx = img.load()
#     pix = np.zeros((width, height, 3), dtype=np.float32) #Numpy version of image
    pix = np.zeros((width + filter_size -1, height + filter_size -1, 3), dtype=np.float32) #Numpy version of image

    
    def valid(r,c, w,h):
        if 0 <= r < w and 0<= c < h:
            return True
        else:
            return False
        
    #Copying pixel values to numpy array
    for i in range(width):
        for j in range(height):
            pix[i + filter_size // 2, j + filter_size // 2] = pixx[i,j]
            
    # HORIZONTAL CONVOLUTION
#     print('First convolution started')
    for i in range(0 , width):
        for j in range(0 , height):
            cPos = (i,j)
            tmp = pix[i :i + (filter_size) ,j]
            tmp = tmp * gauss_filter_x
            tmp = tmp.sum(axis=0)
            pixx[i,j] = (tmp[0], tmp[1], tmp[2])
            
            
    img.save('2-1.png')
#     print("Sucessfully created horizontally convoluted image and saved as 2-1.png")
    
    #Copying pixel values to numpy array
    for i in range(width):
        for j in range(height):
            pix[i + filter_size // 2, j + filter_size // 2] = pixx[i,j]
    
    #VERTICAL CONVOLUTION
#     print('Second convolution started')
    for i in range(0 , width):
        for j in range(0, height):
            cPos = (i,j)
            tmp = pix[i, cPos[1]:cPos[1] + filter_size]
            tmp = tmp * gauss_filter_x
            tmp = tmp.sum(axis=0)
            pixx[i,j] = (tmp[0], tmp[1], tmp[2])
    img.save('2.png')
#     print("Sucessfully created final image and saved as 2.png")
    print("Time taken(sec):", time.time() -t1)


# In[169]:



SeparableGaussianBlurImage('hw1_data/Seattle.jpg', 4)


# In[171]:


def FirstDerivImage(image, sigma):
    t1 = time.time()
    filter_size = 2 * int(sigma * 4 + 0.5) + 1
    print(f'filter size: {filter_size}')
    
    #Creating Filter
    gauss_filter = np.zeros((filter_size, filter_size, 3), dtype=np.float32)
    
    for i in range(filter_size):
        for j in range(filter_size):
            x = i - filter_size // 2
            y = j - filter_size // 2
            tmp = (-1 * x) / (2 * np.pi * sigma ** 4) * np.exp(-(x ** 2 + y ** 2)/(2 * sigma ** 2))
            gauss_filter[i, j] = np.array([tmp, tmp, tmp])
#             gauss_filter_2d[i,j] = tmp
    
#     print(gauss_filter.sum())
    #Reading image using import image
    img = Image.open(image)
    width = img.width
    height = img.height
    pixx = img.load()
#     pix = np.zeros((width, height, 3), dtype=np.float32) #Numpy version of image
    pix = np.zeros((width + filter_size -1, height + filter_size -1, 3), dtype=np.float32)
    
        
    #Copying pixel values to numpy array
    for i in range(width):
        for j in range(height):
            pix[i + filter_size // 2, j + filter_size // 2] = pixx[i,j]
            
    
    #CONVOLUTION
    print('convolution started')
    for i in range(0 , width ):
        for j in range(0, height):
            cPos = (i,j)
            tmp = pix[cPos[0] :cPos[0] + filter_size, cPos[1] :cPos[1] + (filter_size)]
            tmp = tmp * gauss_filter
            tmp = tmp.sum(axis=0)
            tmp = tmp.sum(axis=0)
#             pixx[i,j] = (int(tmp[0] + 64), int(tmp[1] +64), int(tmp[2] + 64))
            pixx[i,j] = (int(tmp[0] + 128), int(tmp[1] + 128), int(tmp[2] + 128))
            #The below option works well because it has high dynamic range than above.
            #In this case brighten to see the image
#             pixx[i,j] = (abs(int(tmp[0])), abs(int(tmp[1])), abs(int(tmp[2])))

    img.save('3a.png')
    print("Sucessfully created and saved as 3a.png")
    print("Time Taken(sec):", time.time()-t1)


# In[172]:


FirstDerivImage('hw1_data/LadyBug.jpg', 1)


# In[174]:


def SecondDerivImage(image, sigma):
    t1 = time.time()
    filter_size = 2 * int(sigma * 4 + 0.5) + 1
#     print(f'filter size: {filter_size}')
    
    #Creating Filter
    gauss_filter = np.zeros((filter_size, filter_size, 3), dtype=np.float32)
    
    for i in range(filter_size):
        for j in range(filter_size):
            x = i - filter_size // 2
            y = j - filter_size // 2
#             tmp = (y * x) / (2 * np.pi * sigma ** 6) * np.exp(-(x ** 2 + y ** 2)/(2 * sigma ** 2))
            tmp = (-1 + (x**2 / sigma**2)) * ( (np.exp(-(x ** 2 + y ** 2)/(2 * sigma ** 2)))/ ( 2 * np.pi * sigma ** 4 ))
            gauss_filter[i, j] = np.array([tmp, tmp, tmp])
#             gauss_filter_2d[i,j] = tmp


    #Creating Filter
    gauss_filter2 = np.zeros((filter_size, filter_size, 3), dtype=np.float32)
    
    for i in range(filter_size):
        for j in range(filter_size):
            x = i - filter_size // 2
            y = j - filter_size // 2
            tmp = (-1 + (y**2 / sigma**2)) * ( (np.exp(-(x ** 2 + y ** 2)/(2 * sigma ** 2)))/ ( 2 * np.pi * sigma ** 4 ))
            gauss_filter2[i, j] = np.array([tmp, tmp, tmp])
    
    
    #Reading image using import image
    img = Image.open(image)
    width = img.width
    height = img.height
    pixx = img.load()
    pix = np.zeros((width + filter_size -1, height + filter_size -1, 3), dtype=np.float32) #Numpy version of image
  
        
    #Copying pixel values to numpy array
    for i in range(width):
        for j in range(height):
            pix[i + filter_size // 2, j + filter_size // 2] = pixx[i,j]
            
    
    #CONVOLUTION
#     print('convolution started')
    for i in range(0 , width):
        for j in range(0 , height):
            cPos = (i,j)
            tmp = pix[cPos[0] :cPos[0] + (filter_size), cPos[1]:cPos[1] + (filter_size)]
            tmp = tmp * gauss_filter
            tmp = tmp.sum(axis=0)
            tmp = tmp.sum(axis=0)
#             pixx[i,j] = (int(tmp[0] + 64), int(tmp[1] +64), int(tmp[2] + 64))
            pixx[i,j] = (int(tmp[0] ), int(tmp[1] ), int(tmp[2]))
            #The below option works well because it has high dynamic range than above.
            #In this case brighten to see the image clearly
#             pixx[i,j] = (abs(int(tmp[0])), abs(int(tmp[1])), abs(int(tmp[2])))


#CONVOLUTION
#     print('convolution started')
    for i in range(0 , width):
        for j in range(0 , height):
            cPos = (i,j)
            tmp = pix[cPos[0] :cPos[0] + (filter_size), cPos[1]:cPos[1] + (filter_size)]
            tmp = tmp * gauss_filter2
            tmp = tmp.sum(axis=0)
            tmp = tmp.sum(axis=0)
#             pixx[i,j] = (int(tmp[0] + 64), int(tmp[1] +64), int(tmp[2] + 64))
            pixx[i,j] =  (int(tmp[0] + pixx[i,j][0] + 128 ), int(tmp[1]+ pixx[i,j][1] + 128 ), int(tmp[2] + pixx[i,j][2] + 128))
            #The below option works well because it has high dynamic range than above.
            #In this case brighten to see the image clearly
#             pixx[i,j] = (abs(int(tmp[0])), abs(int(tmp[1])), abs(int(tmp[2])))

    rpix = np.zeros((width, height, 3), dtype=np.float32)
    for i in range(width):
        for j in range(height):
            rpix[i,j] = pixx[i,j]

    img.save('3b.png')
    print("Sucessfully created and saved as 3b.png")
    print("Time Taken(sec):", time.time()-t1)
    return rpix - 128


# In[176]:


SecondDerivImage('hw1_data/LadyBug.jpg', 1)
print( " ")


# In[177]:


def SharpenImage(image, sigma, alpha):
    t1 = time.time()
    #Reading image using import image
    img1 = Image.open(image)
    width = img1.width
    height = img1.height
    pixx = img1.load()
    pix = np.zeros((width, height, 3), dtype=np.float32)
    
    #Copying pixel values to numpy array
    for i in range(width):
        for j in range(height):
            pix[i,j] = pixx[i,j]
    
    pix2 = SecondDerivImage(image, sigma)
    result = (alpha *pix2)
    result = pix - result
#     print("result dimension", result.shape)
#     print("pix dimensions", pix.shape)
    
    
    #Copying pixel values to numpy array
    for i in range(width):
        for j in range(height):
            pixx[i,j] = (result[i,j][0], result[i,j][1], result[i,j][2])
    
    img1.save('4.png')
    print("Sucessfully created and saved as 4.png")
    print("Time Taken(sec):", time.time()-t1)


# In[178]:


SharpenImage('hw1_data/Yosemite.png', 1, 5)


# In[36]:


def SobelImage(image):
    t1 = time.time()
    #Reading image using import image
    img1 = Image.open(image).convert('LA')
    width = img1.width
    height = img1.height
    pixx = img1.load()
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    
    img = mpimg.imread(image)     
    gray = rgb2gray(img)
    gray = gray * 255
    

    
    #Creating Filter
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter_size = 3
    
    pixx1 = np.zeros((height, width),  dtype=np.float32)
    pixx2 = np.zeros((height, width),  dtype=np.float32)
    orient = np.zeros((height, width, 3), dtype=np.float32)
    
    pix = gray
    print("gray shape",gray.shape)
    print("pixxShape", pixx1.shape)
#     print(pix.shape, pix[333, 1])
#     print(width, height)
    #CONVOLUTION
#     print('convolution started')
    for i in range(1, height -1):
        for j in range(1 , width - 1):
            cPos = (i,j)
#             print(i,j)
            tmp = pix[cPos[0] - (filter_size // 2):cPos[0] + (filter_size // 2) +1, cPos[1] - (filter_size // 2):cPos[1] + (filter_size // 2)+1]
            tmp1 = tmp * g_x
            tmp1 = tmp1.sum(axis=0)
            tmp1 = tmp1.sum(axis=0)
            
            tmp2 = tmp * g_y
            tmp2 = tmp2.sum(axis=0)
            tmp2 = tmp2.sum(axis=0)
            
            pixx1[i,j] = (tmp2 ** 2 + tmp1 ** 2) ** 0.5
            if tmp1 == 0 and tmp2 > 0:
                pixx2[i,j] = math.degrees(math.atan(float('inf')))
            elif tmp1 == 0 and tmp2 < 0:
                pixx2[i,j] = math.degrees(math.atan(float('-inf')))
            elif tmp1 == 0 and tmp2 == 0:
                pixx2[i,j] = 1
            elif tmp1 != 0:
                pixx2[i,j] = (math.degrees(math.atan(tmp2/tmp1)))
            else:
                print("you should not see this +++++++++++++++++++++++++++++++++++=")
            
    
#     #creating gaussian to smoothen orientation image
    
#     sigma = 2
#     filter_size = int(sigma * 4 + 0.5) + 1
#     gauss_filter_2d = np.zeros((filter_size, filter_size), dtype=np.float32)
    
    
    
#     for i in range(filter_size):
#         for j in range(filter_size):
#             x = i - filter_size // 2
#             y = j - filter_size // 2
#             tmp = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 )/(2 * sigma ** 2))
#             gauss_filter_2d[i,j] = tmp
        
#     paddedOrientation = np.zeros((height + filter_size -1, width + filter_size -1), dtype=np.float32) #Numpy version of image
    
        
#     #Copying pixel values to make it padded
#     for i in range(height):
#         for j in range(width):
#             paddedOrientation[i + filter_size // 2, j + filter_size // 2] = pixx2[i,j]
      
#     #Convoluting orientation with gaussian
#     for i in range(0 , height ):
#         for j in range(0 , width):
#             cPos = (i,j)
#             tmp = paddedOrientation[cPos[0]:cPos[0] + (filter_size), cPos[1]:cPos[1] + (filter_size)]
#             tmp = tmp * gauss_filter_2d
#             tmp = tmp.sum(axis=0)
#             tmp = tmp.sum(axis=0)
#             pixx2[i,j] = tmp
            
    pixx1 = pixx1 * 255 / pixx1.max()
#     print(pixx2)
 
    for i in range(0, height):
        for j in range(0, width):
#             #Two color combination
#             if -45 <= pixx2[i,j] <= 45:
#                 orient[i,j] = np.array([0, 255, 0])
            
#             elif -90 <= pixx2[i,j] < -45 or 45 < pixx2[i,j] <= 90:
#                 orient[i,j] = np.array([255, 0, 0])
                
#             else:
#                 orient[i,j] = np.array([0, 0,0])
#             #Three color combination
#             if 30 < pixx2[i,j] <= 90:
#                 orient[i,j] = np.array([0,0,255])
#             elif -30 < pixx2[i,j] <= 30:
#                 orient[i,j] = np.array([0, 255, 0])
#             elif -90 <= pixx2[i,j] <= -30:
#                 orient[i,j] = np.array([255, 0, 0])
#             else:
#                 orient[i,j] = np.array([0,0,0])
                
            #Four parts combination
            if -22.5 <= pixx2[i,j] <= 22.5:
                orient[i,j] = np.array([128, 128, 128])
                
            elif 22.5 < pixx2[i,j] <= 67.5:
                orient[i,j] = np.array([0, 0, 255])
                
            elif 67.5 < pixx2[i,j] <= 90 or -90 <= pixx2[i,j] <= -67.5:
                orient[i,j] = np.array([0, 255, 0])
                
            elif -67.5 < pixx2[i,j] <= -22.5:
                orient[i,j] = np.array([255, 0, 0])
                
            else:
                orient[i,j] = np.array([0, 0, 0])
            
    
    tes = Image.fromarray(pixx1)
    tes = tes.convert("RGB")
    fp = open('5a.png', 'wb')
    tes.save(fp)
    
    cv2.imwrite('5b.png', orient)
    return (pixx1, pixx2)


# In[179]:


SobelImage('hw1_data/LadyBug.jpg')
print(" ")


# In[116]:


SobelImage('hw1_data/TightRope.png')
print(" ")


# In[180]:


def BilinearInterpolation(image, x_, y_):
    x = math.floor(x_)
    y = math.floor(y_)
    
    _x = math.ceil(x_)
    _y = math.ceil(y_)
    
    a = x_ - x
    b = y_ - y
    
    img1 = Image.open(image)
    width = img1.width
    height = img1.height
    f = img1.load()
    
    t1 = (1 - a) * (1 - b) * np.array(f[x,y])
    t2 = a * (1 - b) * np.array(f[_x, y])
    t3 = (1 - a) * b * np.array(f[x, _y])
    t4  = a * b * np.array(f[_x, _y])
    
    return t1 + t2 + t3 + t4
    


# In[ ]:





# In[42]:


#Instead of taking image as an argument here numpy array is passed
def BilinearInterpolationCustom(f, x_, y_):
    x = math.floor(x_)
    y = math.floor(y_)
    
    _x = math.ceil(x_)
    _y = math.ceil(y_)
    
    a = x_ - x
    b = y_ - y
    
    t1 = (1 - a) * (1 - b) * f[x,y]
    
    if _x < f.shape[0]:
        t2 = a * (1 - b) * f[_x, y]
    else:
        t2 = 0 * f[0,0]
        
    if _y < f.shape[1]:
        t3 = (1 - a) * b * f[x, _y]
    else:
        t3 = 0 * f[0,0]
        
    if _x < f.shape[0] and _y < f.shape[1]:
        t4  = a * b * f[_x, _y]
    else:
        t4 = 0 * f[0,0]
        
    return t1 + t2 + t3 + t4


# In[102]:


def nearestNeighborInterpolation(f, x_, y_):
    x = math.floor(x_)
    y = math.floor(y_)
    
    _x = math.ceil(x_)
    _y = math.ceil(y_)
    
#     print("NN", x_, y_)
#     print("fshape", f.shape)
    if x < f.shape[0] -1:
        if  0 <= x_ - x <= 0.5:
            r1 = x
        else:
            r1 = x + 1
    else:
        r1 = x -1
        
    if y < f.shape[1] - 1:
#         print("inside")
        if 0 <= y_ - y <= 0.5:
#             print("one")
            r2 = y
        else:
#             print("two")
            r2 = y + 1
    else:
#         print("else")
        r2 = y -1
#     print (r1, r2)
    return f[r1, r2]


# In[ ]:





# In[183]:


def upsample(image, scalefactor):
    
    img1 = Image.open(image)
    width = img1.width
    height = img1.height
    pixx = img1.load()
#     print(width, height)
    sPix = np.zeros((int(width * scalefactor), int(height * scalefactor), 3), dtype=np.float32)
    
    sPix2 = np.zeros((int(width * scalefactor), int(height * scalefactor), 3), dtype=np.float32)
    pix = np.zeros((width, height, 3), dtype=np.float32)
#     print(pix.shape)
    for i in range(width):
        for j in range(height):
            pix[i,j] = pixx[i,j]
            
            
    for i in range(0, int(width * scalefactor)):
        for j in range(0, int(height * scalefactor)):
#             print(i / scalefactor, j/scalefactor)
            newPix = BilinearInterpolationCustom(pix, i / scalefactor, j/scalefactor)
            newPix2 = nearestNeighborInterpolation(pix, i/scalefactor, j/ scalefactor)
            sPix[i,j] = newPix
            sPix2[i,j] = newPix2
        
    cv2.imwrite('6b.png', sPix)
    cv2.imwrite('6a.png', sPix2)


# In[186]:


upsample('hw1_data/Moire_small.jpg', 4)


# In[ ]:





# In[188]:


def FindPeaksImage(image, thres):
    img1 = Image.open(image)
    width = img1.width
    height = img1.height
    pixx = img1.load()
    result = np.zeros((height, width), dtype=np.float32)
#     result = np.zeros((width, height), dtype=np.float32)
    
#     for i in range(width):
#         for j in range(height):
#             pix[i,j] = pixx[i,j]
    
    magnitude , orientation = SobelImage(image)
    
    for i in range(height):
        for j in range(width):
            if magnitude[i,j] < thres:
                magnitude[i,j] = 0
            else:
                o = orientation[i,j]
                e0 = []
                e1 = []
                e0.append(i + math.cos(math.radians(o)))
                e0.append(j + math.sin(math.radians(o)))
                
                e1.append(i + math.cos(math.radians(-o)))
                e1.append(j + math.sin(math.radians(-o)))
                
                e0Magnitude = BilinearInterpolationCustom(magnitude, e0[0], e0[1])
                e1Magnitude = BilinearInterpolationCustom(magnitude, e1[0], e1[1])
                
                if e0Magnitude < magnitude[i,j] > e1Magnitude:
                    result[i,j] = 255
                else:
                    result[i,j] = 0
    cv2.imwrite('7.png', result)
#     cv2.imwrite('peaks/' + str(image)[5] +'.png', result)


# In[189]:


FindPeaksImage('hw1_data/Circle.png', 40)


# In[ ]:





# In[138]:


for i in range(2, 9,2):
    FindPeaksImage('Gogh/' + str(i) + '.png', 40)


# In[ ]:





# In[157]:


for i in range(0, 1):
    img = cv2.imread('Gogh.png', 0)
    rows,cols = img.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),2,1)
#     dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imwrite('rotate1.png', M)

