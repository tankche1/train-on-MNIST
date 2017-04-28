from numpy import *
from numpy import random
from scipy.ndimage import filters
from scipy.misc import imsave
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import numpy as np



def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """ An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
        using the numerical procedure presented in Eq. (11) of A. Chambolle
        (2005). Implemented using periodic boundary conditions 
        (essentially turning the rectangular image domain into a torus!).
    
        Input:
        im - noisy input image (grayscale)
        U_init - initial guess for U
        tv_weight - weight of the TV-regularizing term
        tau - steplength in the Chambolle algorithm
        tolerance - tolerance for determining the stop criterion
    
        Output:
        U - denoised and detextured image (also the primal variable)
        T - texture residual"""
    
    #---Initialization
    m,n = im.shape #size of noisy image

    U = U_init
    Px = im #x-component to the dual field
    Py = im #y-component of the dual field
    error = 1 
    iteration = 0

    #---Main iteration
    while (error > tolerance):
        Uold = U

        #Gradient of primal variable
        LyU = vstack((U[1:,:],U[0,:])) #Left translation w.r.t. the y-direction
        LxU = hstack((U[:,1:],U.take([0],axis=1))) #Left translation w.r.t. the x-direction

        GradUx = LxU-U #x-component of U's gradient
        GradUy = LyU-U #y-component of U's gradient

        #First we update the dual varible
        PxNew = Px + (tau/tv_weight)*GradUx #Non-normalized update of x-component (dual)
        PyNew = Py + (tau/tv_weight)*GradUy #Non-normalized update of y-component (dual)
        NormNew = maximum(1,sqrt(PxNew**2+PyNew**2))

        Px = PxNew/NormNew #Update of x-component (dual)
        Py = PyNew/NormNew #Update of y-component (dual)

        #Then we update the primal variable
        RxPx =hstack((Px.take([-1],axis=1),Px[:,0:-1])) #Right x-translation of x-component
        RyPy = vstack((Py[-1,:],Py[0:-1,:])) #Right y-translation of y-component
        DivP = (Px-RxPx)+(Py-RyPy) #Divergence of the dual field.
        U = im + tv_weight*DivP #Update of the primal variable

        #Update of error-measure
        error = linalg.norm(U-Uold)/sqrt(n*m);
        iteration += 1;

       # print iteration, error

    #The texture residual
    T = im - U
    print 'Number of ROF iterations: ', iteration
    
    return U,T

f=open('denoise_paremeters.txt','r')
str_denoise=f.read()
f.close()

denoise_paremeters=str_denoise.split(',')
#print(denoise_paremeters)
denoise_tolerance=float(denoise_paremeters[0])
denoise_tau=float(denoise_paremeters[1])
denoise_tv_weight=float(denoise_paremeters[2])

im = mpimg.imread('img.jpg')
U,T = denoise(im,im,denoise_tolerance,denoise_tau,denoise_tv_weight)
imsave('img_denoise.jpg',U)



#print(denoise_tolerance)
#print(denoise_tau)
#print(denoise_tv_weight)




'''
im = zeros((500,500))
im[100:400,100:400] = 128
im[200:300,200:300] = 255

im = im + 30*random.standard_normal((500,500))

imsave('synth_ori.pdf',im)

U,T = denoise(im,im,0.07,0.225+0.02,100+3)

G = filters.gaussian_filter(im,10)


imsave('synth_rof.pdf',U)
imsave('synth_gaussian.pdf',G)
'''

