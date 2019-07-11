# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:04:26 2019

@author: Tim

Purpose: test your image 

"""


# Testing your image after you run the NN script (you need the parameters first)!

 # Testing my CKT AI:
import cv2
import numpy as np
import matplotlib.pyplot as plt # For plotting

size = 128

#change the image_test path to whatever image you want to try with
image_test = cv2.imread ("C:/Users/Tim/Documents/Python/Neural_Network/ckt_tamandesa.jpg") # find the path of the image you want to test on
image_plot = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB) # OpenCV is BGR, matplot lib is RGB


image_resized  = cv2.resize(image_test, (size,size))
image_resized = np.array(image_resized)
image_array =  image_resized.reshape(1,-1).T
image_try =  image_array/255.0



A_final , _ = forward_propagate(image_try , parameters)

if(A_final >= 0.5): #Using sigmoid function
    # It is char kuey teow:
    plt.imshow(image_plot) #plot image
    plt.title('Char Kuey Teow DETECTED!') #Title
else:
    # It is not char kuey teow:
    plt.imshow(image_plot) #plot image
    plt.title('This is not Char Kuey Teow') #Title



