# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 20:15:12 2024

@author: Desubicada
"""
import cv2
import numpy as np
class KalmanFilter:
    def __init__(self):    
        self.kf= cv2.KalmanFilter(4,2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0,0,0,1]], np.float32)
    
    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted= self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y
