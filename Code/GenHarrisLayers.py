"""
==============
Tensorflow implementation of the layers proposed in 

[1] Valentin Penaud--Polge, Santiago Velasco-Forero, Jesus Angulo,
    GenHarris-ResNet: A Rotation Invariant Neural Network Based on Elementary Symmetric Polynomials
    9th Scale Space and Variational Methods in Computer Vision, SSVM 2023

Please cite this reference when using this code.
==============
"""


import numpy as np
from scipy.special import eval_hermitenorm
import math
import tensorflow

from tensorflow.python.keras.layers.pooling import GlobalPooling2D
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *

from tensorflow.keras.models import *
import sys


class GaussianDerivativeESPLayer(tensorflow.keras.layers.Layer):

    """
    Layer computing the ESPs feature maps

    Params : order      - int             - maximal order of derivation of the 
                                            Gaussian Derivative kernels.

             sigma      - list of floats  - scale values used to compute Gaussian
                                            derivatives

             sigma_int  - float           - scale value of the Gaussian kernel used to
                                            integrate the generalized structure tensors
    """

    def __init__(self, order, sigma, sigma_int = 1.0, **kwargs):

        super(GaussianDerivativeESPLayer, self).__init__()

        self.order = order
        self.sigma = sigma
        self.sigma_int = sigma_int

    def build(self, input_shape):

        self.inputChannels = input_shape[-1]
        self.kernels = getGaussianDerivativeKernels(self.order, self.sigma, self.inputChannels)
        
        self.DG_int=getDGIntegrator(self.inputChannels, self.sigma_int)


    def call(self, inputs):

        ESPs = getAllESPs(inputs, self.order, len(self.sigma), self.kernels, DGauss_int = self.DG_int)

        return ESPs

    def get_config(self):

        config = super(GaussianDerivativeESPLayer, self).get_config()
        config.update({
          'order':self.order,
          'sigma':self.sigma,
          'sigma_int':self.sigma_int,
        })
        return config
    

class GlobalTopkExtremaPooling2D(tensorflow.keras.layers.Layer):

    """
    Layer extracting the k higest and k lowest values of all feature maps

    Params : k   - int   - number of highest and lowest values extracted
    """

    def __init__(self, k , **kwargs):
        
        super().__init__()

        self.k = k

    def build(self, input_shape):
        
        self.tensor_shape = input_shape

    def call(self, inputs):

        output = []
        
        for i in range(inputs.shape[-1]):
            reshaped_batch = tensorflow.reshape(inputs[:,:,:,i], shape = (-1, inputs.shape[1]*inputs.shape[2]))
            Ktop = tensorflow.math.top_k(reshaped_batch, k = self.k)

            Kbot = tensorflow.math.top_k(-reshaped_batch, k = self.k)

            output.append(Ktop[0])
            output.append(-Kbot[0])
        
        output = tensorflow.concat(output, axis = -1)

        return output

    def get_config(self):

        config = super(GlobalTopkExtremaPooling2D, self).get_config()
        config.update({
          'k' : self.k
        })

        return config
    

def binomialCoeff(k, n):

    res = math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    return(int(res))

def GaussianDerivative(order, x, sigma):

    """
    Description : Code a Gaussian Derivative kernel G_{order}(x, sigma).
                  
    Params : order - int
             x     - Tensor from Tensorflow.meshgrid
             sigma - float (should be positive) 
    Return : A tensor containing a Gaussian Derivative of order 'order' and scale 'sigma'
    """

    HermitePart = ((-1.0/(np.sqrt(2)*sigma))**order)*eval_hermitenorm(order, x/sigma)
    GaussianPart = np.exp(-(x**2)/(2*sigma**2))/(sigma*np.sqrt(2*math.pi))
    return HermitePart*GaussianPart


@tensorflow.function
def getGaussianDerivativeKernels(order, sigma, inputChannels):

    """
    Description : Compute all the Gaussian derivative kernels up to a given order
                  and for several chosen scale values.

    Params : order           - int
             sigma           - list of floats
             inputChannels   - int
    """

    kernels = []
    for i in range(len(sigma)):

        x, y = np.meshgrid(range(-int(3*sigma[i]), int(3*sigma[i]) + 1), range(-int(3*sigma[i]), int(3*sigma[i]) + 1))
        for j in range(order + 1):

            for k in range(j+1):

                DGauss = (sigma[i]**j)*GaussianDerivative(k, x, sigma[i])*GaussianDerivative(j - k, y, sigma[i])
                DGauss = np.expand_dims(np.expand_dims(DGauss, axis = -1), axis = -1)
                DGauss = np.tile(DGauss, (1,1,inputChannels, 1))
                kernels.append(DGauss.astype('float32'))
    return kernels

@tensorflow.function
def getDGIntegrator(inputChannels, sigma):

    """
    Description : Compute all the Gaussian kernel used to integrate the generalized
                  structure tensors.

    Params : sigma           - list of floats
             inputChannels   - int
    """

    x,y = np.meshgrid(range(-int(3*sigma), int(3*sigma) + 1), range(-int(3*sigma), int(3*sigma) + 1))
    DG_int = GaussianDerivative(0, x, sigma)*GaussianDerivative(0, y, sigma)
    DG_int = np.expand_dims(np.expand_dims(DG_int, axis = -1), axis = -1)
    DG_int = np.tile(DG_int, (1,1,inputChannels, 1))
    return DG_int.astype('float32')

@tensorflow.function
def NewtonIdentities(ESP, PowSum):

    """
    Description : Compute the e_k given the previous ESPs and the Power Sums

    Params : ESP        - list of tensors
             PowSum     - list of tensors
    """

    esp = tensorflow.zeros(shape = tensorflow.shape(ESP[0]), dtype = 'float')
    if len(ESP) != len(PowSum):
        print("Problem Newton Identities")
        return esp
    else:
        n = len(ESP)
        for i in range(n):
            esp = tensorflow.math.add(esp, math.pow(-1, i)*tensorflow.math.multiply(PowSum[i], ESP[-(i+1)]))
        esp = tensorflow.math.multiply(esp, 1.0/float(n))
        return esp
    
@tensorflow.function
def getESPk(Mk, order):

    """
    Description : Compute the ESPs of a generalized structure tensor.
    
    Params : Mk      - tensor (matrix of tensors)
             order   - int

    """

    PowSum = []
    ESP = [tensorflow.ones(shape = tensorflow.shape(Mk)[:-2], dtype = 'float32')]
    res = []
    MatPow = Mk
    for i in range(1, order + 2):

        trace = tensorflow.linalg.trace(MatPow)
        
        PowSum.append(trace)
        EleSymPol = NewtonIdentities(ESP, PowSum)

        ESP.append(EleSymPol)
        
        ESP_homogene = tensorflow.math.abs(EleSymPol)
        
        ESP_homogene = tensorflow.math.pow(ESP_homogene+ sys.float_info.epsilon, 1.0/float(i))
        ESP_homogene = tensorflow.math.multiply((10.0)**order, ESP_homogene)

        res.append(ESP_homogene)
        MatPow = tensorflow.linalg.matmul(Mk, MatPow)

    res = tensorflow.stack(res, axis = -1)
    return res

@tensorflow.function
def getAllESPs(Inputs, order, num_sigma, kernels, DGauss_int):

    """
    Description : Given an input tensor of feature maps, this function compute 
                  the corresponding generalized structure tensors up to a given
                  order. ESPs are computed and returned.

    Params: Input      - tensor
            order      - int
            num_sigma  - int
            kernels    - tensor
            DGauss_int - tensor
    """

    ESPs = []   
    counterKernels = 0
    Matrices = []
    for i in range(num_sigma):
        for j in range(order + 1):
            Lj = []
            for k in range(j+1):
                L = tensorflow.nn.depthwise_conv2d(Inputs, kernels[counterKernels], strides=  [1,1,1,1],  padding = 'SAME')
                counterKernels += 1
                binCoeff = float(binomialCoeff(k, j))
                L = L*(binCoeff**(1.0/2))
                Lj.append(L)
            Lj = tensorflow.stack(Lj, axis = -1)
            Mj = tensorflow.einsum('abcdi,abcdj->abcdij',Lj,Lj)
            Mj = tensorflow.reshape(Mj, shape = [-1, Mj.shape[1], Mj.shape[2], Mj.shape[3], Mj.shape[-1]*Mj.shape[-2]])
            M = []
            for k in range(Mj.shape[-1]):
                M.append(tensorflow.nn.conv2d(Mj[:,:,:,:,k], DGauss_int, strides=[1,1, 1,1], padding='VALID'))
            M = tensorflow.concat(M, axis= -1)
            M = tensorflow.reshape(M, [-1, M.shape[1], M.shape[2], j+1, j+1])
            Matrices.append(M)

    for i in range(num_sigma):

        for j in range(order + 1):

            ESPs.append(getESPk(Matrices[i*(order+1) + j],j))

    ESPs = tensorflow.concat(ESPs, axis = -1)
    return ESPs