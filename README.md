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

google colab of the notebook https://colab.research.google.com/drive/1t3UV97kmAM2V3kTtAkM4Xd6udqu1jQWY?usp=sharing




[Detailled proofs] (https://github.com/Penaud-Polge/GenHarrisResNet/blob/main/proofs.pdf)


Reference:

        
        V. Penaud--Polge, S Velasco-Forero, J Angulo,
        GenHarris-ResNet: A Rotation Invariant Neural Network Based on Elementary Symmetric Polynomials,
        Scale Space and Variational Methods in Computer Vision, 2023
        (Oral Presentation)



