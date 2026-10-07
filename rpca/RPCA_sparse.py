from astropy.io import fits
import numpy as np
from time import time
import matplotlib.pyplot as plt
import os
from h5py import File
from scipy.stats import sigmaclip
from astropy.io import fits

from .utils import centriod_psf


from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, vstack, diags
import math



import torch
from time import time
from scipy.sparse import diags
#from C_tools_lib import centriod_galaxy as centriod_psf
tdtype = torch.float32

#GPU driven version

import torch
import numpy as np
import matplotlib.pyplot as plt
import galsim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate
import torch.optim as optim  
from torch.optim.lr_scheduler import StepLR
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#define the initial model
from scipy.interpolate import interp2d  
from astropy.io import fits
import ctypes
import os
from scipy.stats import sigmaclip
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from scipy.ndimage import zoom 
gspars = galsim.GSParams()


device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps") 
if torch.cuda.is_available():
    device = torch.device("cuda")


dtype = 'float32'



def polyfit2d(pos,val,error,tar_pos,polyorder):
    Nstar=pos.shape[0];iNstar=tar_pos.shape[0]
    l0=0
    for i in range(polyorder+1):
        l0+=i+1
    #print("l0=",l0)
    xy=np.zeros((l0,Nstar),dtype='float')
    a=np.zeros((l0,l0),dtype='float')
    b=np.zeros(l0,dtype='float')
    ival=np.zeros(iNstar,dtype='float')
    ixy=np.zeros((l0,iNstar),dtype='float')
    l=0
    for i in range(polyorder+1):
        for j in range(polyorder+1):
            if (i+j)<=polyorder:
                xy[l,:]=pos[:,0]**i*pos[:,1]**j
                l+=1
    #print(l)
    for l in range(l0):
        for f in range(l0):
            a[l][f]=np.sum(xy[l,:]*xy[f,:]*(1./error**2))
        b[l]=np.sum(xy[l,:]*val*(1./error**2))
    poly=np.linalg.solve(a,b)
    l=0
    for i in range(polyorder+1):
        for j in range(polyorder+1):
            if (i+j)<=polyorder:
                ixy[l,:]=tar_pos[:,0]**i*tar_pos[:,1]**j
                l+=1
    ival=np.dot(poly,ixy)
    return(ival)

def fft_int2D(image, Ng1, Ng2, dx, dy):
    '''
    Constructing shift kernel using fft
    '''
    #print(Ng)
    a=np.ones((Ng1,Ng2))
    F_a = np.fft.fft2(a)
    freq1 = np.fft.fftfreq((Ng1))
    freq2 = np.fft.fftfreq((Ng2))
    k = np.meshgrid(freq1, freq2)
    kx,ky = k[0],k[1] # get the frequency of x and y direction
    shift = 0.* F_a
    shift.real += 1.
    F_shift = shift*np.exp(2j*np.pi*(kx*(-dx)+ky*(-dy)))
    F_S_image = np.fft.fft2(image)*F_shift
    S_image = (np.fft.ifft2(F_S_image).real)
    
    return(S_image)

def gaus_estimate(inimage):
    """
    gaus_estimate(image)
    return(mean,sigma)
    """
    image=inimage.copy()
    nx=image.shape[0];ny=image.shape[1];#print("x=%f,y=%f"%(nx,ny))
    #val=np.empty((0,1),float);
    val=[]
    r2min=(min(nx,ny)/2.)**2;#print("r2min=%f"%(r2min))
    for i in range(nx):
        for j in range(ny):
            x=i-nx/2.+0.5;
            y=j-ny/2.+0.5
            c=image[i][j]
            r2=x*x+y*y
            if r2>r2min and c!=0:
                #val=np.append(val,[[c],],axis=0)
                val.append([c])
    val=np.array(val)
    return(np.mean(val),np.std(val))

def gaus_estimate1(image,gain=1):
    nx, ny = image.shape
    r2min = (min(nx, ny) / 2.) ** 2

    # 直接计算满足条件的像素值，避免创建中间数组
    gridx = np.arange(0, nx, 1)
    gridy = np.arange(0, ny, 1)
    x, y = np.meshgrid(gridx - nx / 2. + 0.5, gridy - ny / 2. + 0.5, indexing='ij')
    r2 = x**2 + y**2
    tmp = image[r2 > r2min]  # 直接提取符合条件的像素值
    #mean = np.mean(tmp)
    #sigma = np.std(tmp)

    return (np.mean(tmp), np.std(tmp))


def S2N(inimage,gain=1):
    image=inimage.copy()
    Ng1=image.shape[0]
    Ng2=image.shape[1]
    weight=np.zeros(image.shape,dtype='float')
    mean,sigma=gaus_estimate1(image)
    #print("method=%f,sigma=%f"%(mean,sigma))
    for i in range(Ng1):
        for j in range(Ng2):
            detx=sigma*sigma*gain*gain
            dety=gain*np.fabs(image[i][j])
            if dety<detx : error=sigma
            else : error=np.sqrt(dety+detx)/gain
            #weight[i][j]=1./(error**2)
            weight[i][j]=error
    snr=np.sum(image)/np.sqrt(np.sum(weight**2))
    return(snr)


def S2N1(inimage, gain=1):
    """
    Vectorized version of the S2N function.
    Parameters:
    ----------
    inimage: 2D numpy array
        Input image.
    gain: float
        Gain factor (default is 1).
    
    Returns:
    -------
    snr: float
        Signal-to-noise ratio (S2N).
    """
    image = inimage.copy()
    mean, sigma = gaus_estimate1(image,gain=gain)
    detx = sigma * sigma * gain * gain
    dety = gain * np.fabs(image)
    error = np.where(dety < detx, sigma, np.sqrt(dety + detx) / gain)
    weight = error
    # Compute SNR
    snr = np.sum(image) / np.sqrt(np.sum(weight**2))
    
    return snr

def interp_fft(image,target_Npix,dx,dy,
                 osam=1,
                 space='real'):
    # print(image.shape,target_Npix,dx,dy,osam)
    newf1=np.zeros(image.shape,dtype='float')
    Ng0=image.shape[0];Nge=target_Npix*osam
    dpix=int(((Ng0-Nge)/2));edx=(dx*osam);edy=(dy*osam)
    outI=np.zeros((target_Npix,target_Npix),dtype='float')
    #print(dpix,edx,edy)
    if space == 'real':
        newf1 = fft_int2D(image,Ng0,Ng0,edx,edy)
        newf1 = newf1[dpix:Nge+dpix, dpix:Nge+dpix]
        outI=newf1.reshape(target_Npix, osam, target_Npix, osam).mean(-1).mean(1)
    if space == 'log':
        sign = np.sign(image)
        #plt.subplot(1,2,1)
        log_star = np.log(np.fabs(image)+1.);#plt.imshow(log_star);plt.colorbar();plt.title('log_star')
        #plt.subplot(1,2,2)
        log_star1 = np.log(log_star+1.);#plt.imshow(log_star1);plt.colorbar();plt.title('log_star1');plt.show()
        log_star2 = np.log(log_star1+1.)*sign;
        newf1 = fft_int2D(log_star2,Ng0,Ng0,edx,edy)
        newf1 = newf1[dpix:Nge+dpix, dpix:Nge+dpix]
        sign = np.sign(newf1)
        #plt.subplot(1,3,1)
        #plt.imshow(image);plt.colorbar();plt.title('image')
        #plt.subplot(1,3,2)
        outI = (np.exp(np.fabs(newf1))-1.);#plt.imshow(outI);plt.colorbar();plt.title('outI')
        outI = (np.exp(outI)-1.);
        #plt.subplot(1,3,3)
        outI = (np.exp(outI)-1.)*sign;#plt.imshow(outI);plt.colorbar();plt.title('outI');plt.show()
        outI = outI.reshape(target_Npix, osam, target_Npix, osam).mean(-1).mean(1)
        
    return(outI)


def interp_cubic(image,target_Npix,dx,dy,
                 osam=1,
                 space='real'):
    # print(image.shape,target_Npix,dx,dy,osam)
    newf1=np.zeros(image.shape,dtype='float')
    Ng0=image.shape[0];Nge=target_Npix*osam
    dpix=((Ng0-Nge)/2);edx=(dx*osam);edy=(dy*osam)
    outI=np.zeros((target_Npix,target_Npix),dtype='float')
    #print(dpix,edx,edy)
    if space == 'real':
        for ic in range(Ng0):
            f1 = interpolate.interp1d(range(Ng0), image[ic] ,kind='cubic')
            newx=np.arange(0,Nge,1)+dpix-edx
            newf1[ic][0:Nge]=f1(newx)
        for ic in range(Nge):
            f2 = interpolate.interp1d(range(Ng0), newf1[:,ic], kind='cubic')
            newy=np.arange(0,Nge,1)+dpix-edy
            newf1[0:Nge,ic]=f2(newy)
        newf1 = newf1[0:Nge, 0:Nge]
        outI=newf1.reshape(target_Npix, osam, target_Npix, osam).mean(-1).mean(1)
    if space == 'log':
        sign = np.sign(image)
        log_star = np.log(np.fabs(image)+1.)*sign
        for ic in range(Ng0):
            f1 = interpolate.interp1d(range(Ng0), log_star[ic] ,kind='cubic')
            newx=np.arange(0,Nge,1)+dpix-edx
            newf1[ic][0:Nge]=f1(newx)
        for ic in range(Nge):
            f2 = interpolate.interp1d(range(Ng0), newf1[:,ic], kind='cubic')
            newy=np.arange(0,Nge,1)+dpix-edy
            newf1[0:Nge,ic]=f2(newy)
        newf1 = newf1[0:Nge, 0:Nge]
        sign = np.sign(newf1)
        outI = (np.exp(np.fabs(newf1))-1.)*sign
        outI = outI.reshape(target_Npix, osam, target_Npix, osam).mean(-1).mean(1)
        
    return(outI)


def pca_star(stars,
              spos,
              NPCs,
              gain,
              psf_size,
              ):
    """
    constructing PSF model by using PCA

    Parameters:
    -------
    instars: 3d numpy array
        input PSF star images used to construct principal components
    spos: 2d numpy array
        position of PSF stars
    NPCs: int
        required number of principal components
    gain: float
        CCD gain
    psf_size: int
        the pixel size:psf_size*psf_size of psf model
    SNR_thresh: float
        signal to noise ratio threshold for the star images,which will used in PCs calculation


    Returns:
    ----------
    PCs: 3d numpy array
        the principal components(PCs)
    slecpos: 2d numpy array
        the selected star image positions aranged as spos[ic][0]->x,spos[ic][1]->y,
        this will be used for the PSF field variation modelings
    coeff: 2d numpy array
        coefficients of PCs
    """
    # sub background and algin center
    instars=stars.copy()
    npc = NPCs
    Ng0 = instars.shape[1]
    cent = np.zeros(2, dtype='float')
    pcastar = []
    ctr = []
    #ic = 0
    for star in instars:
        mean, sigma = gaus_estimate1(star)
        star -= mean
        #SNRs = S2N(star, gain=1)
        #if SNRs > SNR_thresh:
        cent[0], cent[1],_  = centriod_psf(star)
        dx = cent[0]+0.5-(Ng0/2.)
        dy = cent[1]+0.5-(Ng0/2.)
        tmp_star = interp_cubic(star, target_Npix=psf_size, dx=-dy, dy=-dx, space='log')
        tmp_star /= np.sum(tmp_star)
        pcastar.append(tmp_star.ravel())
        #ic += 1
    pcastar = np.array(pcastar)
    Nobj = instars.shape[0]
    coeff = np.ones((Nobj, npc, 2), dtype='float')
    pcastar = pcastar.T
    PCs, sigma, v = np.linalg.svd(pcastar)
    for i in range(npc):
        coeff[:, i, 0] = sigma[i]*v.T[:, i]
    PCs = (PCs[:, 0:npc].T).reshape(npc, psf_size, psf_size)
    
    pcastar = pcastar.T
    residual = []
    Ng = int((pcastar.shape[1])**0.5)
    for ic in range(len(pcastar)):
        tmp = pcastar[ic].copy().reshape(Ng,Ng)
        for l in range(npc):
            tmp -= PCs[l]*coeff[ic][l][0]
        residual.append(tmp)
    residual = np.array(residual)
    return (PCs, coeff, residual)

def image2magnify(image, osam):
    Ng1 = image.shape[0]
    Ng2 = image.shape[1]
    magnified = zoom(image, osam, order=3)
    #parameter_model = torch.nn.Parameter(torch.from_numpy(magnified).double().unsqueeze(0).unsqueeze(0))
    #return(parameter_model.detach().numpy()[0][0])
    return(magnified)

def fft_conv(image,psf):
    '''
    convlove image with a psf,usually image size should be larger than psf
    '''
    print(image.shape,psf.shape)
    core=np.zeros(image.shape,dtype='float')
    nx=image.shape[0];ny=image.shape[1]
    px=psf.shape[0];py=psf.shape[1]
    ra=int((nx-px)/2);rb=int((ny-py)/2)
    core[ra:ra+px,rb:rb+py]=psf;
    Fg=np.fft.fft2(image)
    Fp=np.fft.fft2(core)
    Fg*=Fp
    for i in range(Fp.real.shape[0]):
        for j in range(Fp.real.shape[1]):
            l=(i+j+2)%2
            if (l==1) :Fg.real[i][j]*=-1;Fg.imag[i][j]*=-1;
    tmp=np.fft.ifft2(Fg)
    cimage=tmp.real
    return(cimage)



def pre_data(instars,mask,spos,NPCs,osam,gain=1,nbound=4,SNR_thresh=0.):
    stars=instars.copy()
    Nobj=stars.shape[0]
    Ng0=stars.shape[1]
    Mp = Ng0*Ng0
    shifts=[]
    weights=[]
    out_stars=[]
    tmpPC=np.zeros((NPCs,Ng0,Ng0),dtype='float')
    magPCs=[]
    print('PCA')
    print("Ng0,bound:",Ng0,nbound,)
    for ic in range(Nobj):
        mean,sigma=gaus_estimate1(stars[ic]*mask[ic]);#print("mean,sigma,ic:",mean,sigma,ic+1)
        star = stars[ic,:,:]*mask[ic]-mean
        cx,cy,sigma=centriod_psf(star)
        dx = (cx+0.5)-(Ng0/2.)
        dy = (cy+0.5)-(Ng0/2.)
        shifts.append([dx,dy])
        error=errmap(star)
        fsum=np.sum(star*mask[ic])
        star/=fsum
        error/=fsum
        weight=(1./(error**2))*mask[ic]
        weights.append(weight)
        out_stars.append(star)
        
    weights=np.array(weights)
    shifts=np.array(shifts)
    out_stars = np.array(out_stars)
    PCs, coeff, res=pca_star(stars=instars.copy(),
                                 spos=spos,
                                 NPCs=NPCs,
                                 gain=1,
                                 psf_size=Ng0-nbound,)
    #print('coeff.shape:',coeff[:,:,0].shape)
    Ng=PCs.shape[1]
    dpix=int((Ng0-Ng)/2);#print('dpix=%d,Ng=%d,Ng0=%d'%(dpix,Ng,Ng0))
    for l in range(NPCs):
        tmpPC[l,dpix:Ng+dpix,dpix:dpix+Ng]=PCs[l,:,:]
        mag_PC=image2magnify(tmpPC[l],osam=osam)
        magPCs.append(mag_PC)
        '''plt.subplot(1,2,1)
        plt.imshow(PCs[l]);plt.title('PC:%d'%(l+1));
        plt.subplot(1,2,2)
        plt.imshow(magPCs[l]);plt.title('magPC:%d'%(l+1));plt.show()'''
    magPCs=np.array(magPCs)
    #print(magPCs.shape)
    
    
    magPCs = magPCs.reshape(NPCs, Mp*osam*osam)
    weights = weights.reshape(Nobj, Mp)
    out_stars = out_stars.reshape(Nobj, Mp)
    
    return(magPCs,weights,shifts,coeff[:,:,0],out_stars)
        


def Lanczos2d(Ng,dx,dy,n=6):
    '''
    Constructing shift kernel using Lanczos function
    '''
    M = np.arange(Ng)
    x,y = np.meshgrid(M,M)
    x,y = x - (Ng/2-0.5) - dx, y - (Ng/2-0.5) - dy
    lanczos = np.sinc(np.pi*x)*np.sinc(np.pi*x/n)*np.sinc(np.pi*y)*np.sinc(np.pi*y/n)
    lanczos[np.fabs(x) > n] = 0
    lanczos[np.fabs(y) > n] = 0
    lanczos /= np.sum(lanczos)
    
    return(lanczos)




def coeff2psf(spos, coeffs, PCs, gpos, psf_size, degrees, osam=1):
    """
    interpolate the PSF models at target positions, which is provide for get_psf

    Parameters:
    -------
    spos: numpy 2d array
        the selected star image positions, which is the return from get_PC
    coeff: numpy 2d array
        the coefficients and the corresponding errors of PCs, which is the return from get_PC
    PCs: numpy 3d array
        the principal components(PCs), which is the return from get_PC
    gpos: numpy 2d array
        the target positions of the PSF model required
    psf_size: int
        the pixel size:psf_size*psf_size of psf model
    degrees: int
        order of polynomial used to fit the PSF field variations

    Returns:
    ----------
    rPSF: numpy 3d array
        the constructed PSF model array, which are arrange as rPSF[index][x direction][y direction]
    """
    iNstar = gpos.shape[0]
    Nstar = spos.shape[0]
    Ng = PCs.shape[1]
    Nb = PCs.shape[0]
    icoeff = np.zeros((iNstar, Nb), dtype='float')
    for ic in range(Nb):
        icoeff[:, ic] = polyfit2d(spos,
                                   coeffs[:, ic, 0],
                                   coeffs[:, ic, 1],
                                   gpos,
                                   degrees
                                   )
    rPSF = np.zeros((iNstar, Ng, Ng), dtype='float')
    for ic in range(iNstar):
        for ipc in range(Nb):
            rPSF[ic] += PCs[ipc]*icoeff[ic][ipc]
    dpix = int((Ng-psf_size*osam)/2)
    return (rPSF[:, dpix:dpix+psf_size*osam, dpix:dpix+psf_size*osam])

def size(image, center, sigma):
    '''
    estimated the shape parameters of image by using the second brightness moment

    Parameters:
    -------
    image:2d numpy array
        image to estimated
    center: 1d numpy array
        center position of image, which is estimated from centriod_psf()
    sigma: float
        width of the gaussian weight function

    Returns:
    ----------
    e1: float
        the 1st ellipticity component
    e2: float
        the 2nd ellipticity component
    R2: float
        the size of image
    '''
    nx = image.shape[0]
    ny = image.shape[1]
    # print("nx=%d,ny=%d"%(nx,ny))
    W = 0
    R11 = 0
    R22 = 0
    R12 = 0
    # R2 = 0.
    nh = (np.min([nx, ny])*0.5)**2
    # print("nh=%d,cx=%f,cy=%f"%(nh,center[0],center[1]))
    scale = 0.5/(sigma**2)
    for i in range(nx):
        for j in range(ny):
            x = i-center[0]
            y = j-center[1]
            r2 = x**2+y**2
            weight = np.exp(-r2*scale)
            if r2 < nh:
                W = W+image[i][j]*weight
                R11 = R11+x*x*image[i][j]*weight
                R22 = R22+y*y*image[i][j]*weight
                R12 = R12+x*y*image[i][j]*weight
                # f k<10 :print(image[i][j],weight,sigma)
                # k=k+1;
    R11 = R11/W
    R22 = R22/W
    R12 = R12/W
    # print("message:w11=%f,w12=%f,w22=%f,w=%f"%(R11,R12,R22,W))
    e1 = (R11-R22)/(R11+R22)
    e2 = (2.*R12)/(R11+R22)
    # print("message")
    R2 = R11+R22
    # print(e1,e2,R2)
    if R2 <= 0:
        R2 = 0.001
    if R2 <= 0 or np.fabs(e1) >= 1. or np.fabs(e2) >= 1.:
        # raise Exception("R2<=0\n")
        print("R2<=0\n")
        return (e1, e2, R2)
    else:
        return (e1, e2, R2)
    
def write_mult_fits(fitsname,data1=[],data2=[],data3=[],data4=[],data5=[]):
    if type(data2) is list :
        hdu = fits.PrimaryHDU(data1)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fitsname,overwrite=True) 
    if type(data2) is np.ndarray and type(data3) is list:
        hdu = fits.PrimaryHDU(data1)
        hdu1=fits.ImageHDU(data2)
        hdul = fits.HDUList([hdu,hdu1])
        hdul.writeto(fitsname,overwrite=True) 
    if type(data3) is np.ndarray and type(data4) is list:
        hdu = fits.PrimaryHDU(data1)
        hdu1=fits.ImageHDU(data2)
        hdu2=fits.ImageHDU(data3)
        hdul = fits.HDUList([hdu,hdu1,hdu2])
        hdul.writeto(fitsname,overwrite=True)
    if type(data4) is np.ndarray and type(data5) is list:
        hdu = fits.PrimaryHDU(data1)
        hdu1=fits.ImageHDU(data2)
        hdu2=fits.ImageHDU(data3)
        hdu3=fits.ImageHDU(data4)
        hdul = fits.HDUList([hdu,hdu1,hdu2,hdu3])
        hdul.writeto(fitsname,overwrite=True) 
    if type(data5) is np.ndarray:
        hdu = fits.PrimaryHDU(data1)
        hdu1=fits.ImageHDU(data2)
        hdu2=fits.ImageHDU(data3)
        hdu3=fits.ImageHDU(data4)
        hdu4=fits.ImageHDU(data5)
        hdul = fits.HDUList([hdu,hdu1,hdu2,hdu3,hdu4])
        hdul.writeto(fitsname,overwrite=True) 





def RPCA_sparse(instars,inmask,inspos,npc=10,osam=2,regular=(1,1,1,1),niter_max=10,n=6):
    niter=0;
    nmax=1000;
    detchi2=10000;
    stars=instars.copy()
    mask=np.ones(stars.shape,dtype='float')
    spos=inspos.copy()
    Nobj=stars.shape[0]
    Ng0=stars.shape[1]
    init_chi=10000

    indx = np.where(stars==0)
    mask[indx] *= 0
    
    print(stars.shape)
    #generate initial PCs using PCA, and magnify to the target resolutions
    magPCs,weights,Shifts,Coeffs,slecstar=pre_data(stars,mask,spos,npc,osam,nbound=4)
    
    #indx = np.where(stars==0)
    #mask[indx] *= 0
    print('stars.shape:',stars.shape)
    Nobj,Ng0,_ = stars.shape
    mask = mask.reshape((Nobj,Ng0*Ng0))
    #weights = mask
    
    for ic in range(len(slecstar)):
        slecstar[ic] /= np.sum(slecstar[ic])
    
    for l in range(len(magPCs)):
        sums = (np.sum(magPCs[l]**2))**0.5
        magPCs[l]/=sums
    
    #orthogonalization
    for l in range(npc-1):            
        for ll in range(l + 1, npc):
            projection = np.sum(magPCs[l]*magPCs[ll])
            magPCs[ll] -= projection*magPCs[l]
        sums=(np.sum(magPCs[l]**2))**0.5 #normalization
        magPCs[l]/=sums
    #the last one
    sums = (np.sum(magPCs[npc-1]**2))**0.5 #normalization
    magPCs[npc-1] /= sums
    
    print(magPCs.shape)
    for l in range(len(magPCs)):
        plt.imshow(magPCs[l].reshape(Ng0*osam,Ng0*osam));plt.title('magPC:%d'%(l+1));plt.show()
    
    coeffs=Coeffs.copy()
    stars=slecstar.copy()#restar.shape is not the same as stars
    restars=np.zeros(stars.shape,dtype='float')
    chi2=0
    PCs=magPCs.copy()
    #constructing the shift kernels
    print("constructing the downsizing and shift kernels")
    Down_size = create_downsampling_matrix_sparse(Ng0*osam, osam)
    print("create_downsampling_matrix_torch down")
    S_matrix = []
    for ic in range(len(Shifts)):
        t1=time()
        # kernel = S_ma_torch(Ng0*osam, Ng0*osam, Shifts[ic,1]*osam, Shifts[ic,0]*osam, n=20*osam)
        #kernel = S_ma_sparse(Ng0*osam, Ng0*osam, Shifts[ic,1]*osam, Shifts[ic,0]*osam, n=4*osam)
        kernel = S_ma_fully_vectorized(Ng0*osam, Ng0*osam, Shifts[ic,1]*osam, Shifts[ic,0]*osam, n=n)
        sparse_kernel = Down_size @ kernel
        S_matrix.append(sparse_kernel)
        t2=time()
        print(ic," time:",t2-t1)
    print("preparing linear matrix")
    #S_matrix = torch.stack(S_matrix)
    
    ma_list = pre_matrix_numpy(S_matrix, 
                               Down_size,
                               stars,
                               weights,
                               osam,
                               *regular,
                               )
            
    print("starting iterations")
    while niter<niter_max:
        print("PCs.shape:",PCs.shape)
        Coeffs = Solving_Coeff1(stars,weights,PCs,osam,S_matrix)
        PCs = Solving_PCs1(stars,weights,S_matrix,PCs,osam,Coeffs,Down_size,regular,ma_list)
        restars[:,:] *= 0.
        count=0
        chi2=0
        for ic in range(Nobj):
            for l in range(npc):
                SPC = (S_matrix[ic] @ PCs[l])
                restars[ic] += SPC*Coeffs[ic][l]
            #Coeffs[ic,:] /= torch.sum(restars[ic])
            #restars[ic].masked_fill_(restars[ic] < 0, 0) 
            #restars[ic]=restars[ic].abs()
            #restars[ic] /= torch.sum(restars[ic])
            chi2 += np.sum(((stars[ic]-restars[ic])**2)*weights[ic])
        detchi2=np.fabs(init_chi/chi2-1.)
        init_chi=chi2
        niter+=1
        print('niter=%d,detchi2=%f,chi2=%f'%(niter,detchi2,chi2/Nobj/Ng0/Ng0))
        judge_chi=init_chi/Nobj/Ng0/Ng0
        if detchi2<1.0e-8 or chi2<2.:
            break
            
    for l in range(npc):
        plt.imshow(PCs[l].reshape(Ng0*osam,Ng0*osam));plt.title("npc=%d"%(l));plt.show()
    #calculate the variance of coefficients
    #Coeffs = Coeffs.cpu().numpy()
    Coeffs = []
    for ic in range(Nobj):
        shifPC=[]
        for l in range(npc):
            cPC = S_matrix[ic] @ PCs[l]
            shifPC.append(cPC)
        shifPC=np.array(shifPC)
        PtV=np.zeros(shifPC.shape,dtype='float')
        for l in range(npc):
            PtV[l,:] = shifPC[l,:]*weights[ic]
        PtVP = PtV @ shifPC.T
        tmp = PtV @ stars[ic]
        #iPtVP = np.linalg.inv(PtVP)
        #Coeff = iPtVP @ tmp
        coeff = np.linalg.solve(PtVP,tmp)
        '''coeff = solve_with_normalization_and_error(torch.tensor(shifPC.T,dtype=tdtype).to(device), 
                                                   torch.tensor(stars[ic],dtype=tdtype).to(device), 
                                                   torch.tensor(weights[ic],dtype=tdtype).to(device)
                                                  )
        coeff = coeff.cpu().numpy()'''
        #sums = 0
        #for l in range(npc):
        #    sums += np.sum(shifPC[l] * Coeff[l])
        #Coeff /= sums
        
        Coeffs.append(coeff)
    Coeffs=np.array(Coeffs);print('Solving_Coeff,Coeffs.shape',Coeffs.shape)
    # Coeffs = Coeffs.cpu().numpy()
    recoeffs=[]
    residual = []
    restars = np.zeros((Nobj,Ng0*Ng0), dtype='float32')
    for ic in range(Nobj):
        tmp=[]
        for l in range(npc):
            SPC = S_matrix[ic] @ PCs[l]
            Coeff_err=1./np.sum(SPC**2*weights[ic])
            tmp.append([Coeffs[ic][l],Coeff_err])
            restars[ic] += SPC*Coeffs[ic][l]
        recoeffs.append(tmp)
        residual.append(stars[ic]-restars[ic])
    recoeffs=np.array(recoeffs)
    residual = np.array(residual)
    
    return(PCs,recoeffs,magPCs,coeffs,residual)


def solve_with_normalization_and_error(P, I, V_inv):
    #print("P.size(), I.size(), V_inv.size():",P.size(), I.size(), V_inv.size())
    # 计算 A = (P^T V^{-1} P)^{-1} P^T V^{-1}
    sum_P = torch.sum(P,axis=0)
    PtV = P.t() * V_inv
    PtVP = PtV @ P
    A = torch.inverse(PtVP)
    #print("sum_P.size(), A.size(), (I*V_inv).size():",sum_P.size(), A.size(), (I*V_inv).size())
    up = sum_P @ A @ P.t() @ (I*V_inv) - 1
    down = sum_P @ A @ sum_P.t()
    
    lambda_ = 2 * up / down
    
    C = A @ (PtV @ I - lambda_/2 * sum_P.t())
    
    return C


def Solving_Coeff1(stars,weights,PCs,osam,S_ma):
    Nobj, Mp = stars.shape
    npc=PCs.shape[0]
    Coeffs=[]
    #for l in range(npc):
    #    sums = (torch.sum(PCs[l].pow(2)))**0.5
    #    print("PC in Solving_Coeff1 sum is %.5f",sums)
    for ic in range(Nobj):
        #print("sm matrix sum is%.5f"%(torch.sum(S_ma[ic])))
        shifPC=[]
        for l in range(npc):
            PC_tensor = PCs[l]
            pooled_PC_sum = (S_ma[ic] @ PC_tensor)
            shifPC.append(pooled_PC_sum)
        shifPC = np.array(shifPC)
        PtV = np.zeros(shifPC.shape, dtype='float32')
        for l in range(npc):
            PtV[l,:] = shifPC[l,:] * weights[ic,:]
        PtVP = PtV @ shifPC.T 
        tmp = PtV @ stars[ic]
        #iPtVP = torch.linalg.inv(PtVP)
        #Coeff = iPtVP @ tmp
        coeff = np.linalg.solve(PtVP,tmp)
        #print(shifPC.shape)
        #coeff = solve_with_normalization_and_error(shifPC.T, stars[ic], weights[ic])
        #sums = torch.sum(PCs.T @ coeff)
        #coeff /= sums
        #print("normalised flux is %.5f, star flux is %.5f"%(torch.sum(PCs.T @ coeff),torch.sum(stars[ic])))
        Coeffs.append(coeff)
    Coeffs=np.array(Coeffs);print('Solving_Coeff,Coeffs.shape',Coeffs.shape)
    
    return(Coeffs)



def Solving_PCs1(stars,weights,S_matrix,PCs,osam,Coeffs,Down_size,regular, ma_list):
    Nobj, Mp =stars.shape
    Ng0 = int(Mp**0.5)
    npc=PCs.shape[0]
    newPCs=[]
    print('Training PCs')
    substars=stars.copy()
    #for l in range(npc):
    l=0
    while l < npc:
        PC = train_PC1(substars,weights,S_matrix,PCs[l],osam,Coeffs[:,l],Down_size,regular, ma_list)
        t1 = time()
        newPCs.append(PC)
        for i in range(Nobj):
            PCi = S_matrix[i] @ PC  # Move to CPU for numpy conversion
            substars[i] -= PCi * Coeffs[i,l]
        t2 = time()
        # print('iter time:',t2-t1)
        l+=1
    newPCs = np.array(newPCs)
    #orthogonalization
    for l in range(npc-1):            
        for ll in range(l + 1, npc):
            projection = np.sum(newPCs[l]*newPCs[ll])
            newPCs[ll] -= projection*newPCs[l]
        sums=(np.sum(newPCs[l]**2))**0.5 #normalization
        newPCs[l]/=sums
    #the last one
    sums = (np.sum(newPCs[npc-1]**2))**0.5 #normalization
    newPCs[npc-1] /= sums
    return(newPCs)

def train_PC1(instars, weights, S_matrix, PC, osam, Coeff, Down_size, regular, ma_list):
    stars = instars.copy()
    Nobj = stars.shape[0]
    num_epochs = 10
    # print('constructAB_sparse')
    t1 = time()
    A_sparse, b = constructAB_sparse(S_matrix, 
                                    Down_size, 
                                    stars, 
                                    weights, 
                                    PC,
                                    Coeff,
                                    ma_list,
                                    *regular)
    t2 = time()
    print('constructAB_sparse down',t2-t1)
    #P_next = spsolve(A_sparse, b)
    #factor = cholesky(A_sparse)
    #P_next = factor(b)
    A_dense = A_sparse.toarray()
    P_next = np.linalg.solve(A_dense, b)
    norm = math.sqrt(np.dot(P_next, P_next))
    P_next /= norm

    '''P_next = spsolve(A_sparse, b)
    norm = math.sqrt(np.dot(P_next, P_next))
    P_next /= norm'''
    t3 = time()
    print('solving down',t3-t2)
    return (P_next) 



def Lanczos_numpy(Ng, dx, dy, n=6, device='cpu'):
    '''
    Args:
        Ng (int): 核尺寸 (Ng x Ng)
        dx (float): x 方向偏移量
        dy (float): y 方向偏移量
        n (int): Lanczos 窗截断参数，默认6
        device (str): 张量设备 ('cpu' 或 'cuda')
    Returns:
        lanczos (Tensor): 生成的 Lanczos 核 (Ng, Ng)
    '''
    # 生成网格坐标 (对齐 NumPy 的 xy 索引模式)
    M = np.arange(Ng)
    y, x = np.meshgrid(M, M, indexing='ij')  # PyTorch 默认 ij 索引，需转 xy
    
    # 坐标中心化并添加偏移量
    center = Ng / 2.0
    x = x - center - dx
    y = y - center - dy
    
    # 计算 Lanczos 函数
    x_pi = np.pi * x
    y_pi = np.pi * y
    
    # 计算 sinc 项 (注意 PyTorch 的 sinc 定义与 NumPy 一致: sin(πx)/(πx))
    sinc_x = np.sinc(x_pi)  # 等价于 sin(πx)/(πx)
    sinc_x_window = np.sinc(x_pi / n)  # 窗口函数
    
    sinc_y = np.sinc(y_pi)
    sinc_y_window = np.sinc(y_pi / n)
    
    # 组合 Lanczos 核
    lanczos = (sinc_x * sinc_x_window) * (sinc_y * sinc_y_window)
    
    # 应用截断条件 (|x| > n 或 |y| > n 的位置置零)
    mask = (np.fabs(x) <= n) & (np.fabs(y) <= n)
    lanczos = lanczos * mask
    
    # 归一化
    lanczos_sum = np.sum(lanczos)
    if lanczos_sum != 0:  # 避免除以零
        lanczos /= lanczos_sum
    
    return lanczos

def create_downsampling_matrix_sparse(Ng, osam):
    '''
    Create a downsampling matrix using sparse matrix representation.
    Each row corresponds to a block of size (osam x osam) in the original grid.
    '''
    ng = Ng // osam  # 下采样后的尺寸
    
    # 用于构建稀疏矩阵的数据结构
    rows = []
    cols = []
    data = []

    row_idx = 0  # 行索引，从 0 到 ng*ng - 1

    for i in range(ng):
        for j in range(ng):
            # 对应的 box 在原图中的起始位置
            start_row = i * osam
            start_col = j * osam

            # 遍历该 box 内的所有像素
            for di in range(osam):
                for dj in range(osam):
                    img_row = start_row + di
                    img_col = start_col + dj
                    
                    if img_row < Ng and img_col < Ng:
                        # 每个 box 中的像素位置对应 B 矩阵中的一列
                        col_idx = img_row * Ng + img_col
                        # 每个 box 对应 B 的一行，这一行中有 osam^2 个 1
                        rows.append(row_idx)
                        cols.append(col_idx)
                        data.append(1.0)
            row_idx += 1

    # 构造稀疏矩阵：形状 (ng*ng, Ng*Ng)
    shape = (ng * ng, Ng * Ng)

    # 使用 COO 格式创建稀疏矩阵
    B_sparse = coo_matrix((data, (rows, cols)), shape=shape)

    # 推荐转换为 CSR 格式以提高后续乘法性能
    return B_sparse.tocsr()


def Lanczos(Ng,dx,dy,n=6):
    '''
    Constructing shift kernel using Lanczos function
    '''
    M = np.arange(Ng)
    x,y = np.meshgrid(M,M)
    x,y = x - (Ng/2) - dx, y - (Ng/2) - dy
    lanczos = np.sinc(np.pi*x)*np.sinc(np.pi*x/n)*np.sinc(np.pi*y)*np.sinc(np.pi*y/n)
    lanczos[np.fabs(x) > n] = 0
    lanczos[np.fabs(y) > n] = 0
    lanczos /= np.sum(lanczos)
    
    return(lanczos)



def S_ma_optimized(Ngi, Ng, dx, dy, n=6):
    '''
    Construct shift matrix by iterating over kernel elements and vectorizing index calculations.
    '''
    kernel2d = Lanczos(Ng, dx, dy, n)  # 确保Lanczos函数正确生成核
    #kernel2d = fft_kernel(Ng, dx, dy)
    #indx = np.where(np.fabs(kernel2d)<1.e-5)
    #kernel2d[indx] *= 0
    nhalf = Ng // 2
    ngi = Ngi
    
    # 初始化用于存储非零元素的数据结构
    rows = []
    cols = []
    data = []

    # 获取所有非零核元素的索引
    non_zero = np.argwhere(kernel2d != 0)

    for is_, js in non_zero:
        kernel_value = kernel2d[is_, js]
        # 计算当前核元素对应的位移
        dx_k = is_ - nhalf
        dy_k = js - nhalf
        
        # 计算有效的i范围
        i_min = max(0, -dx_k)
        i_max = min(ngi-1, ngi-1 - dx_k)
        if i_min > i_max:
            continue
        
        # 计算有效的j范围
        j_min = max(0, -dy_k)
        j_max = min(ngi-1, ngi-1 - dy_k)
        if j_min > j_max:
            continue
        
        # 生成i和j的坐标网格
        i, j = np.ogrid[i_min:i_max+1, j_min:j_max+1]
        
        # 计算对应的输入位置
        it = i + dx_k
        jt = j + dy_k
        
        # 计算一维索引
        row_idx = (it * ngi + jt).ravel()
        col_idx = (i * ngi + j).ravel()

        # 添加数据
        rows.append(row_idx)
        cols.append(col_idx)
        data.append(np.full_like(row_idx, kernel_value, dtype=np.float32))

    # 合并所有数据
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    # 创建稀疏矩阵
    shape = (ngi * ngi, ngi * ngi)
    tmp_sparse = coo_matrix((data, (rows, cols)), shape=shape)
    
    return tmp_sparse



def S_ma_sparse(Ngi, Ng, dx, dy, n=20):
    '''
    使用 NumPy 实现带有周期性核的移位矩阵，并返回稀疏矩阵形式
    '''
    # 生成 Lanczos 核
    kernel2d = Lanczos(Ng, dx, dy, n)  # [Ng, Ng]
    
    ngi = Ngi
    nhalf = Ng // 2
    
    # 创建索引网格 (NumPy 实现)
    i, j = np.meshgrid(np.arange(ngi), np.arange(ngi), indexing='ij')
    it, jt = np.meshgrid(np.arange(ngi), np.arange(ngi), indexing='ij')

    # 计算相对位移
    x = i[:, :, np.newaxis, np.newaxis] - it[np.newaxis, np.newaxis, :, :]
    y = j[:, :, np.newaxis, np.newaxis] - jt[np.newaxis, np.newaxis, :, :]

    # 应用周期性边界条件
    is_mod = (x + nhalf) % Ng
    js_mod = (y + nhalf) % Ng

    # 收集非零元素的位置和值
    rows = []
    cols = []
    data = []

    for i in range(ngi*ngi):
        for j in range(ngi*ngi):
            if kernel2d[is_mod.flat[i], js_mod.flat[j]] != 0:
                rows.append(i)
                cols.append(j)
                data.append(kernel2d[is_mod.flat[i], js_mod.flat[j]])

    # 构建稀疏矩阵
    tmp_sparse = coo_matrix((data, (rows, cols)), shape=(ngi*ngi, ngi*ngi))

    return tmp_sparse.tocsr()  # 返回 CSR 格式的稀疏矩阵，更高效地进行矩阵运算


def fft_kernel(Ng, dx, dy):
    '''
    使用 numpy 实现基于 FFT 的位移核生成
    '''
    # 生成坐标网格 (PyTorch 实现)
    M = np.arange(Ng)
    x, y = np.meshgrid(M, M, indexing='ij')
    x = x - Ng//2  # 中心化坐标
    y = y - Ng//2

    # 创建全1矩阵并计算 FFT
    a = np.ones((Ng, Ng), dtype='float32')
    F_a = np.fft.fft2(a)

    # 生成频率网格 (与 NumPy 的 fftfreq 等效)
    freq = np.fft.fftfreq(Ng)
    kx, ky = np.meshgrid(freq, freq, indexing='ij')  # [Ng, Ng]

    # 构建位移相位因子
    shift = F_a*0.
    shift.real += 1.
    phase = 2j * np.pi * (kx * (-dy) + ky * (-dx))
    F_shift = shift * np.exp(phase)  # 位移频域响应

    # 逆变换获取核函数
    kernel = np.fft.ifft2(F_shift).real
    kernel = np.fft.fftshift(kernel)  # 与 NumPy 的 fftshift 等效


    # 归一化处理
    kernel /= kernel.sum()

    return kernel


def create_dx(size, rows, cols):
    Dx = diags([-1.0, 1.0], [0, 1], shape=(size, size), format='csr')
    for i in range(1, rows):
        Dx[i * cols - 1, i * cols] = 0  # 去除行末到下一行首的连接
    return Dx

def create_dy(size, rows, cols):
    Dy = diags([-1.0, 1.0], [0, cols], shape=(size, size), format='csr')
    return Dy

def create_d_d1(size, rows, cols):
    Dd1 = diags([-1.0, 1.0], [0, cols + 1], shape=(size, size), format='csr')
    for i in range(1, rows):
        Dd1[i * cols - 1, i * cols] = 0
    return Dd1

def create_d_d2(size, rows, cols):
    Dd2 = diags([-1.0, 1.0], [0, cols - 1], shape=(size, size), format='csr')
    for i in range(1, rows):
        Dd2[i * cols, i * cols - 1] = 0
    return Dd2

def create_dxx(size, rows, cols):
    Dxx = diags([-2.0, 1.0, 1.0], [0, 1, 2], shape=(size, size), format='csr')
    for i in range(1, rows):
        Dxx[i * cols - 2, i * cols] = 0
        Dxx[i * cols - 1, i * cols + 1] = 0
    return Dxx

def create_dyy(size, rows, cols):
    Dy = diags([-2.0, 1.0, 1.0], [0, cols, 2 * cols], shape=(size, size), format='csr')
    return Dy

def create_dd1d1(size, rows, cols):
    Dd1 = diags([-2.0, 1.0, 1.0], [0, cols+1, 2*(cols+1)], shape=(size, size), format='csr')
    for i in range(1, rows):
        Dd1[i * cols - 2, i * cols] = 0
        Dd1[i * cols - 1, i * cols + 1] = 0
    return Dd1

def create_dd2d2(size, rows, cols):
    Dd2 = diags([-2.0, 1.0, 1.0], [0, cols-1, 2*(cols-1)], shape=(size, size), format='csr')
    for i in range(1, rows):
        Dd2[i * cols - 2, i * cols] = 0
        Dd2[i * cols - 1, i * cols + 1] = 0
    return Dd2


def SGD_solve(A, b):
    x = torch.zeros_like(b, requires_grad=True).to(device)
    optimizer = optim.SGD([x], lr=0.0001)
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = torch.sum((A @ x - b)**2)
        loss.backward()
        optimizer.step()
        
    return(x.detach())


def constructAB_sparse(DS, D, I, sigma, P_init, Coeff,
                                  ma_tuple,
                                  lambda_L2=1e-3, lambda_L1=0, lambda_TV=0, lambda_grad=0):
    """
    完全向量化构造稀疏矩阵 A 和稠密向量 b。
    
    参数:
        DS, D, I, sigma, P_init, Coeff: 数据输入
        ma_tuple: 包含多个稀疏矩阵的元组 (W_DS, W_DS_DS, W_DS_I, ma_L2, ma_grad, ma_TV, DTD)
        各种正则化参数
    
    返回:
        A: 稀疏矩阵 (csr_matrix)
        b: 稠密向量 (np.ndarray)
    """
    # 解包 ma_tuple
    W_DS, W_DS_DS_list, W_DS_I_list, ma_L2, ma_grad, ma_TV, DTD = ma_tuple

    num_obs = len(Coeff)
    Mpe = P_init.shape[0]

    # 将 W_DS_DS_list 转换为稀疏矩阵堆叠格式
    if isinstance(W_DS_DS_list, list):
        W_DS_DS = vstack(W_DS_DS_list)  # 形状变为 (num_obs*Mpe, Mpe)，但这不是最终形式
    else:
        W_DS_DS = W_DS_DS_list

    coeff_squared = Coeff ** 2
    # 方法二：更简单的方法 —— 直接遍历所有 W_DS_DS[i] 并加权相加
    A = csr_matrix((Mpe, Mpe), dtype='float32')
    for i in range(num_obs):
        A += W_DS_DS_list[i].multiply(coeff_squared[i])

    # 构造 b：向量化版本
    b = np.zeros(Mpe, dtype=np.float32)
    if isinstance(W_DS_I_list, list):
        for i in range(num_obs):
            vec = W_DS_I_list[i].toarray().flatten()
            b += vec * Coeff[i]
    else:
        # 假设 W_DS_I_list 是一个大稀疏矩阵 (num_obs, Mpe)
        b += W_DS_I_list.multiply(Coeff[:, None]).sum(axis=0).A1

    # **1. L2 正则化**
    if lambda_L2 > 0 and ma_L2 is not None:
        A +=  ma_L2

    # **2. L1 正则化（稀疏性）**
    if lambda_L1 > 0:
        diag_L1 = np.abs(P_init.ravel())
        A +=  diags(diag_L1, format='csr')

    # **3. Total Variation 正则化**
    if lambda_TV > 0 and ma_TV is not None:
        A +=  ma_TV

    # **4. 梯度正则化**
    if lambda_grad > 0 and ma_grad is not None:
        A +=  ma_grad

    # **5. 先验项**
    if DTD is not None:
        A += DTD

    return A, b



def constructAB_sparse_fast(DS, D, I, sigma, P_init, Coeff,
                             ma_tuple,
                             lambda_L2=1e-3, lambda_L1=0, lambda_TV=0, lambda_grad=0):
    """
    快速构造稀疏矩阵 A 和稠密向量 b。
    假设 W_DS_DS_list 和 W_DS_I_list 已经是稀疏矩阵列表。
    """
    # 解包 ma_tuple
    W_DS, W_DS_DS_list, W_DS_I_list, ma_L2, ma_grad, ma_TV, DTD = ma_tuple

    num_obs = len(Coeff)
    Mpe = P_init.shape[0]

    # 初始化 A 和 b
    A = csr_matrix((Mpe, Mpe), dtype='float32')
    b = np.zeros(Mpe, dtype=np.float32)

    coeff_squared = Coeff ** 2

    # 构造 A 和 b
    for i in range(num_obs):
        A += W_DS_DS_list[i].multiply(coeff_squared[i])  # 数据项 A
        vec = W_DS_I_list[i]
        b[vec.indices] += vec.data * Coeff[i]  # 数据项 b

    # 正则化项
    if lambda_L2 > 0 and ma_L2 is not None:
        A += ma_L2

    if lambda_L1 > 0:
        diag_L1 = np.abs(P_init.ravel())
        A += diags(diag_L1, format='csr')

    if lambda_TV > 0 and ma_TV is not None:
        A += ma_TV

    if lambda_grad > 0 and ma_grad is not None:
        A += ma_grad

    if DTD is not None:
        A += DTD

    return A, b




def first_order_diff_matrix(n, shape):
    """
    生成一阶差分矩阵（水平、垂直、两条对角线方向）
    """
    rows, cols = shape
    size = rows * cols  

    # 1. 水平梯度 Dx
    Dx = diags([-1, 1], [0, 1], shape=(size, size)).toarray()
    for i in range(1, rows):  
        Dx[i * cols - 1, i * cols] = 0  

    # 2. 垂直梯度 Dy
    Dy = diags([-1, 1], [0, cols], shape=(size, size)).toarray()

    # 3. 左上到右下对角梯度 D_d1
    Dd1 = diags([-1, 1], [0, cols + 1], shape=(size, size)).toarray()
    for i in range(1, rows):  
        Dd1[i * cols - 1, i * cols] = 0  

    # 4. 右上到左下对角梯度 D_d2
    Dd2 = diags([-1, 1], [0, cols - 1], shape=(size, size)).toarray()
    for i in range(1, rows):  
        Dd2[i * cols, i * cols - 1] = 0  

    # 组合多个方向梯度
    D_first = np.vstack([Dx, Dy, Dd1, Dd2])
    
    return D_first

def second_order_diff_matrix(n, shape):
    """
    生成二阶差分矩阵（水平、垂直、两条对角线方向）
    """
    rows, cols = shape
    size = rows * cols  

    # 1. 水平二阶差分 Dxx
    Dxx = diags([1, -2, 1], [0, 1, 2], shape=(size, size)).toarray()
    for i in range(1, rows):  
        Dxx[i * cols - 2, i * cols] = 0  
        Dxx[i * cols - 1, i * cols + 1] = 0  

    # 2. 垂直二阶差分 Dyy
    Dyy = diags([1, -2, 1], [0, cols, 2 * cols], shape=(size, size)).toarray()

    # 3. 左上到右下对角线二阶差分 Dd1d1
    Dd1d1 = diags([1, -2, 1], [0, cols + 1, 2 * (cols + 1)], shape=(size, size)).toarray()
    for i in range(1, rows):  
        Dd1d1[i * cols - 2, i * cols] = 0  
        Dd1d1[i * cols - 1, i * cols + 1] = 0  

    # 4. 右上到左下对角线二阶差分 Dd2d2
    Dd2d2 = diags([1, -2, 1], [0, cols - 1, 2 * (cols - 1)], shape=(size, size)).toarray()
    for i in range(1, rows):  
        Dd2d2[i * cols - 2, i * cols] = 0  
        Dd2d2[i * cols - 1, i * cols + 1] = 0  

    # 组合多个方向的二阶差分
    D_second = np.vstack([Dxx, Dyy, Dd1d1, Dd2d2])
    
    return D_second


def periodic_padding(kernel, padding):
    """
    对二维张量进行周期性填充。
    :param kernel: 输入的二维张量 [H, W]
    :param padding: 填充大小（假设上下左右填充相同）
    :return: 周期性填充后的张量 [H+2*padding, W+2*padding]
    """
    H, W = kernel.shape
    padded_kernel = torch.zeros((H + 2 * padding, W + 2 * padding), dtype=kernel.dtype, device=kernel.device)

    # 中心部分：原始核
    padded_kernel[padding:H + padding, padding:W + padding] = kernel

    # 左右填充
    padded_kernel[padding:H + padding, :padding] = kernel[:, -padding:]  # 左侧
    padded_kernel[padding:H + padding, -padding:] = kernel[:, :padding]  # 右侧

    # 上下填充
    padded_kernel[:padding, :] = padded_kernel[-2 * padding:-padding, :]  # 上方
    padded_kernel[-padding:, :] = padded_kernel[padding:2 * padding, :]   # 下方

    # 四角填充
    padded_kernel[:padding, :padding] = kernel[-padding:, -padding:]      # 左上角
    padded_kernel[:padding, -padding:] = kernel[-padding:, :padding]      # 右上角
    padded_kernel[-padding:, :padding] = kernel[:padding, -padding:]      # 左下角
    padded_kernel[-padding:, -padding:] = kernel[:padding, :padding]      # 右下角

    return padded_kernel


def zero_padding(kernel, padding):
    """
    对二维张量进行零填充。
    :param kernel: 输入的二维张量 [H, W]
    :param padding: 填充大小（假设上下左右填充相同）
    :return: 零填充后的张量 [H+2*padding, W+2*padding]
    """
    return F.pad(kernel.unsqueeze(0).unsqueeze(0), 
                 pad=(padding, padding, padding, padding), 
                 mode='constant', value=0).squeeze()


def size1(image, center, sigma):
    nx, ny = image.shape
    nrow, ncol = np.arange(0, nx), np.arange(0, ny)
    x, y = np.meshgrid(ncol, nrow)
    cx, cy = x - center[0], y - center[1]
    r2 = cx**2 + cy**2
    scale = 0.5 / (sigma**2)
    weight = np.exp(-r2 * scale)
    
    # 添加权重范围限制
    nh = (np.min([nx, ny]) * 0.5)**2
    mask = r2 < nh
    
    # 加权图像
    W_im = weight * image
    
    # 归一化因子
    W = np.sum(W_im[mask])
    
    # 计算 R11, R22, R12
    R11 = np.sum(W_im[mask] * (cx[mask]**2)) / W
    R22 = np.sum(W_im[mask] * (cy[mask]**2)) / W
    R12 = np.sum(W_im[mask] * (cx[mask] * cy[mask])) / W
    
    # 计算椭圆参数
    e1 = (R11 - R22) / (R11 + R22)
    e2 = (2.0 * R12) / (R11 + R22)
    R2 = R11 + R22
    
    # 检查异常值
    if R2 <= 0 or np.fabs(e1) >= 1.0 or np.fabs(e2) >= 1.0:
        print("R2<=0\n")
        return (e1, e2, R2)
    else:
        return (e1, e2, R2)



def pre_matrix_numpy(S_matrix, Down_size, stars, weights, osam = 2,
                     lambda_L2=1e-3, lambda_L1=0, lambda_TV=0, lambda_grad=0, 
                     center_smooth_radius=5, smooth_strength=10):
    """
    使用 NumPy + SciPy 稀疏矩阵实现预计算矩阵
    """
    num_obs = len(S_matrix)
    Mpe = S_matrix[0].shape[1]  # 参数空间大小
    Mp = S_matrix[0].shape[0]   # 观测空间大小
    print("Mpe,Mp:",Mpe,Mp)

    W_DS = []       # List of (Mpe, Mp)
    W_DS_DS = []    # List of (Mpe, Mpe)
    W_DS_I = []     # List of (Mpe, )

    for i in range(num_obs):
        weight_i = csr_matrix(diags(weights[i]))  # (Mp, Mp)
        W_DS_i = S_matrix[i].T @ weight_i         # (Mpe, Mp)
        W_DS.append(csr_matrix(W_DS_i))

        # W_DS_DS_i = W_DS_i @ S_matrix[i]
        W_DS_DS_i = W_DS_i @ S_matrix[i]  # (Mpe, Mpe)
        W_DS_DS.append(csr_matrix(W_DS_DS_i))

        I_i = stars[i]  # (Mp,)
        W_DS_I_i = W_DS_i @ I_i  # (Mpe,)
        W_DS_I.append(csr_matrix(W_DS_I_i))

    # 构造 DTD = Down_size.T @ Down_size
    DTD = Down_size.T @ Down_size  # (Mpe, Mpe)
    DTD = csr_matrix(DTD)


    height = width = int(np.sqrt(Mpe))

    # 创建距离权重
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    center_y, center_x = height // 2 - 0.5, width // 2 - 0.5
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    distance_weight = np.clip(distance - center_smooth_radius, a_min=0, a_max=None)**1.5
    distance_weight = distance_weight.flatten()

    if lambda_L2>0:
        ma_L2 = lambda_L2 * np.eye(Mpe, dtype='float32')  # 控制 ΔP 大小
        ma_L2 = csr_matrix(ma_L2)
    else:
        ma_L2 = 0
    
    if lambda_TV>0:
        L = second_order_diff_matrix(Mpe, (int(np.sqrt(Mpe)), int(np.sqrt(Mpe))))
        weighted_L = L * distance_weight[None, :]  # 对行进行加权
        ma_TV = lambda_TV * (weighted_L.T @ weighted_L)  #二阶差分矩阵
        ma_TV = csr_matrix(ma_TV)
    else:
        ma_TV = 0
    
    if lambda_grad>0:
        D_grad = first_order_diff_matrix(Mpe, (int(np.sqrt(Mpe)), int(np.sqrt(Mpe))))
        weighted_D_grad = D_grad * distance_weight[None, :]  # 对行进行加权
        ma_grad = lambda_grad * (weighted_D_grad.T @ weighted_D_grad) #一阶差分矩阵
        ma_grad = csr_matrix(ma_grad)
    else:
        ma_grad = 0
    

    return (
        W_DS,
        W_DS_DS,
        W_DS_I,
        ma_L2,
        ma_grad,
        ma_TV,
        DTD
    )

def S_ma_fully_vectorized(Ngi, Ng, dx, dy, n=6):
    '''
    完全向量化的实现 (适用于中等规模矩阵)
    '''
    kernel = Lanczos(Ng, dx, dy, n)  # 假设Lanczos函数已正确定义
    #kernel = fft_kernel(Ng, dx, dy)
    indx = np.where(np.fabs(kernel)<1.e-5)
    kernel[indx] *= 0
    nhalf = Ng // 2
    
    # 初始化用于存储非零元素的数据结构
    rows = []
    cols = []
    data = []

    # 获取所有非零核元素的索引
    non_zero = np.argwhere(kernel != 0)

    for is_, js in non_zero:
        kernel_value = kernel[is_, js]
        
        # 计算当前核元素对应的位移
        di = is_ - nhalf
        dj = js - nhalf
        
        # 计算有效的i和j范围
        i_min = max(0, -di)
        i_max = min(Ngi-1, Ngi-1 - di)
        if i_min > i_max:
            continue
        
        j_min = max(0, -dj)
        j_max = min(Ngi-1, Ngi-1 - dj)
        if j_min > j_max:
            continue
        
        # 生成i和j的坐标网格
        i, j = np.ogrid[i_min:i_max+1, j_min:j_max+1]
        
        # 计算对应的输入位置
        it = i + di
        jt = j + dj
        
        # 计算一维索引
        row_idx = (it * Ngi + jt).ravel()
        col_idx = (i * Ngi + j).ravel()

        # 添加数据
        rows.append(row_idx)
        cols.append(col_idx)
        data.append(np.full_like(row_idx, kernel_value, dtype=np.float32))

    # 合并所有数据
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    # 创建稀疏矩阵
    shape = (Ngi * Ngi, Ngi * Ngi)
    tmp_sparse = coo_matrix((data, (rows, cols)), shape=shape)
    
    return tmp_sparse

def errmap(image,gain=1.):
    Ng1=image.shape[0]
    Ng2=image.shape[1]
    errors=np.zeros(image.shape,dtype='float')
    mean,sigma=gaus_estimate1(image)
    #print("method=%f,sigma=%f"%(mean,sigma))
    for i in range(Ng1):
        for j in range(Ng2):
            detx=sigma*sigma*gain*gain
            dety=gain*np.fabs(image[i][j])
            if dety<detx : error=sigma
            else : error=np.sqrt(dety+detx)/gain
            #weight[i][j]=1./(error**2)
            errors[i][j]=error
    return(errors)

