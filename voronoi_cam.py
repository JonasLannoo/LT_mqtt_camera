# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:13:00 2022

@author: u0110583
"""

import numpy as np
import cv2
import shutil

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

#%% Defining MQTT settings

import paho.mqtt.client as paho

broker="mqtt.devbit.be"
port=1883

client1 = paho.Client("PanelController")
client1.connect(broker,port)

#%%

# Draw voronoi diagram
def draw_voronoi(img, facets, indices) :
    # ( facets, centers) = subdiv.getVoronoiFacetList([]) 
    colors = []
    for i in range(len(facets)) :
    # for i in indices:
        # ifacet_arr = []
        # for f in facets[i] :
        #     ifacet_arr.append(f)
        # ifacet = np.array(ifacet_arr, np.int)
        ifacet = facets[i].astype(np.int32)
        # color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        # create mask for polygon
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask,[ifacet.astype(np.int32)],(255))
        values = frame[np.where(mask == 255)]
        # print(values.shape)
        color = (np.average(values,axis=0))
        cor_color = (np.power(color/255,2.4)*255).astype(int)
        # cor_color = color.astype(int)
        colors.append(cor_color)
        # print("0x%02X%02X%02X, "%(cor_color[1],cor_color[0],cor_color[2]))
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        # ifacets = np.array([ifacet])
        # cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        # cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)
        
    cstring = []
    for i in np.argsort(indices):
        cor_color = colors[i]
        # print("0x%02X%02X%02X, "%(cor_color[1],cor_color[0],cor_color[2]))
        cstring.append(cor_color[2])
        cstring.append(cor_color[1])
        cstring.append(cor_color[0])
        
    ret = client1.publish("LannooPanel/2x2RGBvalues",bytearray(cstring))
    print(ret)

#%%
import time

# define a video capture objectq
vid = cv2.VideoCapture(cv2.CAP_DSHOW) # 0 as argument
ret, frame = vid.read()

# or an image
# frame = imageio.imread("test_fig.png")

# Rectangle to be used with Subdiv2D
size = frame.shape

dsubx = 107.76
dsuby = 109.03
dx = 3*dsubx
dy = 3*dsuby

prim_unit = np.array([[83.08, 368.57],
                      [116.65, 353.16],
                      [154.98, 345.04],
                      [80.74, 329.51],
                      [134.55, 314.05],
                      [62.96, 290.33],
                      [101.12, 298.58],
                      [137.07, 275.03]])

panel = np.zeros((9*prim_unit.shape[0],2))
for i in range(3):
    for j in range(3):
        idx = prim_unit.shape[0]*(i+j*3)
        panel[idx:idx+prim_unit.shape[0],:] = prim_unit+np.tile([i*dsubx, -j*dsuby],[prim_unit.shape[0],1])

# sort by x cord
panel = panel[np.argsort(panel[:,0]),:]
unique_xs = np.unique(panel[:,0])
for xs in unique_xs:
    idxs = np.where(panel[:,0]==xs)[0]
    jdxs = np.argsort(panel[idxs,1])
    panel[np.min(idxs):np.min(idxs)+len(idxs),1] = panel[idxs[jdxs],1]

screen = np.zeros((4*panel.shape[0],2))
for i in range(2):
    for j in range(2):
        idx = panel.shape[0]*(i+j*2)
        screen[idx:idx+panel.shape[0],:] = panel+np.tile([i*dx, (1-j)*dy],[panel.shape[0],1])
        
x0 = np.min(screen[:,0])
y0 = np.min(screen[:,1])

screen = screen-np.array([x0,y0])

xm = np.max(screen[:,0])
ym = np.max(screen[:,1])

scale = np.min([size[0]/ym,size[1]/xm])*0.9

screen = screen*scale

xt = np.abs(np.max(screen[:,0])-size[1])/2
yt = np.abs(np.max(screen[:,1])-size[0])/2

screen = screen+np.array([xt,yt])

# panel_LED_indexes = np.array([46, 22, 0, 61, 44, 20, 62, 45, 21, 47, 23, 1, 63, 43, 19, 60, 42, 18, 48, 24, 2, 64, 41, 17, 49, 25, 3, 59, 40, 16, 65, 50, 26, 58, 39, 4, 66, 38, 15, 57, 37, 14, 51, 27, 5, 67, 36, 13, 56, 28, 6, 68, 35, 12, 69, 52, 29, 55, 30, 7, 70, 34, 11, 54, 32, 9, 53, 31, 8, 71, 33, 10], dtype=int)
panel_LED_indexes = np.array([ 0, 22, 46, 20, 44, 61, 21, 45, 62,  1, 23, 47, 19, 43, 63, 18, 42,
       60,  2, 24, 48, 17, 41, 64,  3, 25, 49, 16, 40, 59, 26, 50, 65,  4,
       39, 58, 15, 38, 66, 14, 37, 57,  5, 27, 51, 13, 36, 67,  6, 28, 56,
       12, 35, 68, 29, 52, 69,  7, 30, 55, 11, 34, 70,  9, 32, 54,  8, 31,
       53, 10, 33, 71])

screen_LED_indexes = np.zeros(4*panel_LED_indexes.shape[0],dtype=int)
for i in range(4):
    screen_LED_indexes[panel_LED_indexes.shape[0]*i:panel_LED_indexes.shape[0]*i+panel_LED_indexes.shape[0]] = panel_LED_indexes+panel_LED_indexes.shape[0]*i

# points = np.random.randint(40,460,(500,2))
points = screen

rect = (0, 0, size[1], size[0])
# Create an instance of Subdiv2Dq
subdiv = cv2.Subdiv2D(rect);
# Insert points into subdiv
for p in points :
    subdiv.insert(tuple(p))
    
# Allocate space for Voronoi Diagram
img_voronoi = np.zeros(frame.shape, dtype = frame.dtype)
( facets, centers) = subdiv.getVoronoiFacetList([]) 
while(True):
      
    # Capture the video frame
    # by frame
    start = time.time()

    ret, frame = vid.read()
    # frame = imageio.imread("test_fig.png")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    image_height, image_width, _ = frame.shape

    
    
    # Draw Voronoi diagram
    # draw_voronoi(img_voronoi,subdiv)
    print("{")
    draw_voronoi(img_voronoi,facets, screen_LED_indexes)
    print("};")
    # Display the resulting frame
    
    for fdx,facet in enumerate(facets):
        x,y = np.mean(facet,axis=0)
        img_voronoi = cv2.putText(img_voronoi, "%d.%d"%(screen_LED_indexes[fdx]//72,screen_LED_indexes[fdx]%72), (x.astype(int),y.astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2)
    
    cv2.imshow('frame', np.concatenate((img_voronoi, frame, ),axis=1))
    
    end = time.time()
    seconds = end - start
    fps  = 1 / seconds
    # print("Estimated frames per second : {0}".format(fps))
    
    
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choiceqq
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

