import sys
import cv2
import numpy as np
from numpy import genfromtxt
from keras import backend as K
import tensorflow as tf
from align import AlignDlib
from model import create_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches

THRESHOLD = 1.03

FRmodel = create_model()
FRmodel.load_weights('openface/nn4.small2.v1.h5')
alignment = AlignDlib('models/landmarks.dat')

def image_compare(pair):
    """
    Comparison of image pairs with face recognition

    Arguments:
    pair -- tuple containing two filename strings

    Returns:
    none
    """
    print "[*] Comparing images {} and {}".format(pair[0], pair[1])

    # try to extract faces from images
    img1 = extract_face(load_image(pair[0]))
    img2 = extract_face(load_image(pair[1]))

    if img1 is None:
        print "[!] Face recognition failed for {}".format(pair[0])
    if img2 is None:
        print "[!] Face recognition failed for {}".format(pair[1])
    if img1 is None or img2 is None:
        return None

    # calculate the distance and print it
    dist = check_distance(img1, img2)
    if dist <= THRESHOLD:
        result_str = "\033[0;32mMATCH\033[0m"
    else:
        result_str = "\033[0;31mMISMATCH\033[0m"
    print "[*] Distance is {} [{}]".format(dist, result_str)

    # write the result into the output file
    open("pairs_out.txt","a+").write(pair[0]+'\t'+pair[1]+'\t'+str(distance)+'\n')

    return dist

def show_pair(pair, dist):
    (img1, img2) = pair

    if dist is None:
        result_str = "?? [ERROR]"
        font = {'color': 'purple'}
    elif dist > THRESHOLD:
        result_str = "{:.2f} [MISMATCH]".format(dist)
        font = {'color': 'red'}
    else:
        result_str = "{:.2f} [MATCH]".format(dist)
        font = {'color': 'green'}

    plt.figure(figsize=(8,3))
    plt.suptitle("Distance = {}".format(result_str), fontdict=font)
    
    # load img1
    img1 = load_image(pair[0])
    plt.subplot(121)
    plt.imshow(img1)
    bb = alignment.getLargestFaceBoundingBox(img1)
    if bb:
        plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))
    
    # load img2
    img2 = load_image(pair[1])
    plt.subplot(122)
    plt.imshow(img2)
    bb = alignment.getLargestFaceBoundingBox(img2)
    if bb:
        plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

    plt.show()

def check_distance(img1, img2):
    return distance(predict(img1), predict(img2))

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def predict(img):
    img = (img / 255.).astype(np.float32)
    return FRmodel.predict(np.expand_dims(img, axis=0))[0]

def load_image(fileName):
    return cv2.imread("lfw/{}".format(fileName))[...,::-1]

def extract_face(image):
    return alignment.align(96, image, alignment.getLargestFaceBoundingBox(image),\
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def parse_pairs(fileName):
    """
    Parse pairs from the given file

    Arguments:
    fileName -- name of the file containing the pairs

    Returns:
    pairs -- list containing tuples
    """
    pairs = []
    f = open(fileName)
    f.readline() # first line is useless

    for line in f.readlines():
        l = line.rstrip().split()

        # if there are 3 columns, its a match
        if len(l) == 3:
            pairs.append(("{}/{}_{:04d}.jpg".format(l[0], l[0], int(l[1])),\
                            "{}/{}_{:04d}.jpg".format(l[0], l[0], int(l[2]))))
        
        # otherwise it must be a 4 column entry, which is a mismatch
        elif len(l) == 4:
            pairs.append(("{}/{}_{:04d}.jpg".format(l[0], l[0], int(l[1])),\
                            "{}/{}_{:04d}.jpg".format(l[2], l[2], int(l[3]))))

        else:
            print "[!] Unrecognized line"

    return pairs

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "batch":
        pairs = parse_pairs(sys.argv[2])
        for pair in pairs:
            image_compare(pair)
    elif cmd == "compare":
        pair = ("_".join(sys.argv[2].split('_')[:-1])+"/"+sys.argv[2],\
                "_".join(sys.argv[3].split('_')[:-1])+"/"+sys.argv[3])
        dist = image_compare(pair)
        show_pair(pair, dist)
    else:
        usage()