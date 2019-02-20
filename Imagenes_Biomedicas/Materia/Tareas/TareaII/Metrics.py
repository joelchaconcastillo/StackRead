import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

def Sensitivity(TP, FN):
   return  TP/(TP+FN)
def Specifity(TN, FP):
   return TN/(TN+FP)
def Accuracy(TP, TN, FP,FN):
   return (TP+TN)/(TP + FP + TN + FN)
def MTCC(TP, TN, FP, FN):
   return  ((TP*TN)-(FP*FN))/math.sqrt(  (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
def JACCARD_INDEX(TP, FP, FN):
   return TP/(FP+FN+TP)
