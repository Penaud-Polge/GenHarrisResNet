# GenHarris-ResNet: A Rotation Invariant Neural Network Based on Elementary Symmetric Polynomials


<img src="/figures/scheme.png" alt="GanHarris" style="width:750px;"/>

GenHarris-ResNet architecture: The first part of the proposed network
is equivariant to rotation and is composed of several feature function blocks
arranged in a ResNet configuration. Then, the feature maps are given to the
global extrema pooling layer which brings the invariance property. A dense layer
ends the network.

Code:

Example of pretrained model on fashion mnist is given in:
https://github.com/Penaud-Polge/GenHarrisResNet/blob/main/Code/test_GenHarris-ResNet.ipynb




[Detailled proofs] (https://github.com/Penaud-Polge/GenHarrisResNet/blob/main/proofs.pdf)


Reference:

        
        V. Penaud--Polge, S Velasco-Forero, J Angulo
        Scale Space and Variational Methods in Computer Vision, 2023
        (Oral Presentation)



