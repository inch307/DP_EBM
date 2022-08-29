import warnings
import numpy as np
warnings.filterwarnings('ignore', message='overflow encountered in exp')
class A():
    def __init__(self):
        

        print(np.exp(1000000000))
        print(1+1/(1+np.exp(100000000)))