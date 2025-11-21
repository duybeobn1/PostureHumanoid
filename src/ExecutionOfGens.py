import os
import cv2
import sys
import numpy as np

from VideoSkeleton import *
from GenNearest import *
from Skeleton import *


def run_gennearest_test():
    """ Fonction pour tester la logique du plus proche voisin. """
    
    force = False
    modFRame = 100
    filename = "../data/taichi1.mp4" 
    
    s = VideoSkeleton(filename, force, modFRame, 400, 1.0, False)
    print(s)

    generator = GenNearest(s)
    
    index_test = 50 

    if s.skeCount() <= index_test:
        index_test = 0

    if s.skeCount() == 0:
        print("\n--- Échec du Test ---")
        print("La base de données de squelettes est vide.")
        return

    print(f"\n--- Lancement du test GenNeirest avec l'index {index_test} ---")
    
    test_ske = s.ske[index_test]
    original_image = s.readImage(index_test)
    
    generated_image = generator.generate(test_ske)
    
    # Affichage de Original vs. Généré
    if generated_image is not None and original_image is not None:
        img_original_copy = original_image.copy()
        img_generated_copy = generated_image.copy()

        test_ske.draw(img_original_copy) 
        test_ske.draw(img_generated_copy) 
        
        result_display = combineTwoImages(img_original_copy, img_generated_copy)
        
        print("Affichage du résultat : Image Originale (gauche) vs. Image NN Générée (droite).")
        cv2.imshow('Resultat Nearest Neighbor (Original vs. Genere)', result_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("L'image générée ou originale n'a pas pu être chargée. Vérifiez les chemins d'accès.")


if __name__ == '__main__':
    run_gennearest_test()
    