
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNearest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Nearest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        """ generator of image from skeleton """
        
        min_distance = float('inf')
        best_match_index = -1

        for i in range(self.videoSkeletonTarget.skeCount()):
            current_ske = self.videoSkeletonTarget.ske[i]
            
            distance = ske.distance(current_ske)
            
            if distance < min_distance:
                min_distance = distance
                best_match_index = i

        if best_match_index != -1:
            best_image = self.videoSkeletonTarget.readImage(best_match_index)
            print(f"Nearest Neighbor trouvé à l'index {best_match_index} à une distance {min_distance:.4f}")
            return best_image
        else:
            print("Erreur: Aucun squelette trouvé dans la cible.")
            empty = np.zeros((64,64, 3), dtype=np.uint8) # Image noire d'erreur
            return empty




