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
import h5py
from typing import List, Union
from scipy.signal import fftconvolve
from ctypes import c_float, c_int, POINTER

import os
import sys

# 动态添加当前目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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



def save_nested_dict_to_hdf5(filename, data, group='/', f=None):
    """
    递归地将嵌套字典保存到 HDF5 文件中。
    
    :param filename: HDF5 文件名
    :param data: 要保存的数据（字典）
    :param group: 当前组的路径
    :param f: 已打开的 HDF5 文件对象（用于递归调用）
    """
    if f is None:
        with h5py.File(filename, 'w') as f:
            save_nested_dict_to_hdf5(filename, data, group, f)
        return
    
    for key, value in data.items():
        full_path = f"{group}/{key}"
        if isinstance(value, dict):
            subgroup = f.create_group(full_path)
            save_nested_dict_to_hdf5(filename, value, full_path, f)
        else:
            try:
                # 尝试直接保存为数据集
                f.create_dataset(full_path, data=value)
            except TypeError:
                # 如果是字符串或其他不可序列化类型，尝试转换为字符串保存
                f.create_dataset(full_path, data=str(value))


def load_nested_dict_from_hdf5(filename, group='/', f=None):
    """
    递归地从 HDF5 文件中加载嵌套字典。
    
    :param filename: HDF5 文件名
    :param group: 当前组的路径
    :param f: 已打开的 HDF5 文件对象（用于递归调用）
    :return: 加载的嵌套字典
    """
    if f is None:
        with h5py.File(filename, 'r') as f:
            return load_nested_dict_from_hdf5(filename, group, f)
    
    result = {}
    for key in f[group].keys():
        full_path = f"{group}/{key}"
        if isinstance(f[full_path], h5py.Group):
            result[key] = load_nested_dict_from_hdf5(filename, full_path, f)
        else:
            dataset = f[full_path][()]
            # 尝试将字符串类型的数据集转换回原始类型
            if isinstance(dataset, bytes):
                dataset = dataset.decode('utf-8')
            result[key] = dataset
    return result

def get_filename(
    file_dir: str,
    match: Union[str, List[str], None] = None,
    avoid: Union[str, List[str], None] = None
) -> List[str]:
    """
    获取指定目录下符合匹配条件且不包含排除条件的文件名。

    参数:
    - file_dir: 目录路径
    - match: 单个字符串或字符串列表，文件名必须包含所有指定的字符串
    - avoid: 单个字符串或字符串列表，文件名不能包含任何指定的字符串

    返回:
    - 符合条件的文件名列表
    """
    # 确保 match 和 avoid 是列表
    if match is None:
        match_list = []
    elif isinstance(match, str):
        match_list = [match]
    else:
        match_list = list(match)

    if avoid is None:
        avoid_list = []
    elif isinstance(avoid, str):
        avoid_list = [avoid]
    else:
        avoid_list = list(avoid)

    fname = []
    for filename in os.listdir(file_dir):
        file_path = os.path.join(file_dir, filename)
        if os.path.isfile(file_path):
            # 检查是否包含所有匹配字符串
            if all(m in filename for m in match_list):
                # 检查是否不包含任何排除字符串
                if not any(a in filename for a in avoid_list):
                    fname.append(filename)

    return fname




def write_mult_fits(fitsname, *args):
    """
    自动将任意数量的数组写入 FITS 文件。
    
    参数:
    fitsname: 输出文件名
    *args: 可变数量的 numpy 数组 (data1, data2, ...)
    """
    if not args:
        print("警告: 没有传入数据。")
        return

    hdu_list = []
    
    # 1. 处理第一个数据 (必须是 PrimaryHDU)
    # 假设 args[0] 对应原来的 data1
    hdu_list.append(fits.PrimaryHDU(args[0]))
    
    # 2. 自动处理剩余的数据 (作为 ImageHDU 追加)
    # 从 args[1] 开始遍历到最后
    for data in args[1:]:
        hdu_list.append(fits.ImageHDU(data))
    
    # 3. 写入文件
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(fitsname, overwrite=True)
    
    # 可选：打印确认信息
    # print(f"成功写入 {len(args)} 个数据层到 {fitsname}")

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
        # mean, sigma = gaus_estimate1(star)
        # star -= mean
        #SNRs = S2N(star, gain=1)
        #if SNRs > SNR_thresh:
        cent[0], cent[1],_ = centriod_psf(star)
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



def pre_data(instars,mask,spos,NPCs,osam,gain=1,nbound=4,SNR_thresh=0.):
    stars=instars.copy()
    Nobj=stars.shape[0]
    Ng0=stars.shape[1]
    Mp = Ng0*Ng0
    shifts=[]
    weights=[]
    out_stars=[]
    errors = []
    tmpPC=np.zeros((NPCs,Ng0,Ng0),dtype='float')
    magPCs=[]
    print('PCA')
    print("Ng0,bound:",Ng0,nbound,)
    for ic in range(Nobj):
        # mean,sigma=gaus_estimate1(stars[ic]*mask[ic]);#print("mean,sigma,ic:",mean,sigma,ic+1)
        star = (stars[ic,:,:])*mask[ic]
        #star = (stars[ic,:,:]-mean)*mask[ic]
        cx,cy,_=centriod_psf(star)
        dx = (cx+0.5)-(Ng0/2.)
        dy = (cy+0.5)-(Ng0/2.)
        shifts.append([dx,dy])
        error=errmap(star)
        fsum=np.sum(star*mask[ic])
        star/=fsum
        error/=fsum
        errors.append(error)
        weight=(1./(error**2))*mask[ic]
        weights.append(weight)
        out_stars.append(star)
        
    weights=np.array(weights)
    shifts=np.array(shifts)
    errors = np.array(errors)
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
    
    return(magPCs,weights,shifts,coeff[:,:,0],out_stars,errors)



def size(image,center,sigma):
    '''
    size(image,center,sigma)
    center[0]=cx;center[1]=cy
    '''
    nx=image.shape[0];ny=image.shape[1]
    W=0;R11=0;R22=0;R12=0;R2=0.;k=0;
    nh=(np.min([nx,ny])*0.5)**2
    scale=0.5/(sigma**2)
    for i in range(nx):
        for j in range(ny):
            x=i-center[0];y=j-center[1]
            r2=x**2+y**2;weight=np.exp(-r2*scale)
            if r2<nh:
                W=W+image[i][j]*weight
                R11=R11+x*x*image[i][j]*weight
                R22=R22+y*y*image[i][j]*weight
                R12=R12+x*y*image[i][j]*weight

                
    R11=R11/W;R22=R22/W;R12=R12/W

    e1=(R11-R22)/(R11+R22)
    e2=(2.*R12)/(R11+R22)
    #print("message")
    R2=R11+R22
    #print(e1,e2,R2)
    if R2<=0 : R2=0.001
    if R2<=0 or np.fabs(e1)>=1. or np.fabs(e2)>=1.:
        #raise Exception("R2<=0\n")
        print("R2<=0\n");
        return(e1,e2,R2)
    else:
        return(e1,e2,R2)
    
def shear2vector(g1,g2):
    mage=np.sqrt(g1**2+g2**2)
    sin=g2/mage;cos=g1/mage
    Tant=sin/cos
    theta=np.arctan2(sin,cos)
    theta/=2.
    e1=mage*np.cos(theta);e2=mage*np.sin(theta)
    return(e1,e2)
        
def periodic_extension(image, pad_width=5):
    """
    对图像进行周期性扩展，每边扩展pad_width个像素。
    使用FFT中的周期性边界条件，即右侧超出的部分用左侧填充，反之亦然。
    """
    Ng0 = image.shape[0]
    
    # 创建扩展后的图像数组
    extended_image = np.zeros((Ng0 + 2 * pad_width, Ng0 + 2 * pad_width), dtype=image.dtype)
    
    # 中心部分直接赋值为原图
    extended_image[pad_width:-pad_width, pad_width:-pad_width] = image
    
    # 上下边界扩展
    for i in range(pad_width):
        extended_image[i, pad_width:-pad_width] = image[-pad_width + i, :]  # 上边界
        extended_image[-1 - i, pad_width:-pad_width] = image[i, :]  # 下边界
    
    # 左右边界扩展
    for i in range(pad_width):
        extended_image[:, i] = extended_image[:, -2 * pad_width + i]  # 左边界
        extended_image[:, -1 - i] = extended_image[:, pad_width + i]  # 右边界
    
    # 四个角的扩展（确保四个角也符合周期性）
    extended_image[:pad_width, :pad_width] = image[-pad_width:, -pad_width:]  # 左上角
    extended_image[:pad_width, -pad_width:] = image[-pad_width:, :pad_width]  # 右上角
    extended_image[-pad_width:, :pad_width] = image[:pad_width, -pad_width:]  # 左下角
    extended_image[-pad_width:, -pad_width:] = image[:pad_width, :pad_width]  # 右下角
    
    return extended_image
        
        
def interp_cubic_ex(inimage,target_Npix,dx,dy,
                 osam=1,
                 space='real',
                 kind = 'cubic'):
    """
    kind : str or int, optional
        Specifies the kind of interpolation as a string or as an integer
        specifying the order of the spline interpolator to use.
        The string has to be one of 'linear', 'nearest', 'nearest-up', 'zero',
        'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero',
        'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
        zeroth, first, second or third order; 'previous' and 'next' simply
        return the previous or next value of the point; 'nearest-up' and
        'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5)
        in that 'nearest-up' rounds up and 'nearest' rounds down. Default
        is 'linear'.
    
    """
    # print(image.shape,target_Npix,dx,dy,osam)
    pad_width = 10*osam
    image = periodic_extension(inimage, pad_width)
    newf1=np.zeros(image.shape,dtype='float')
    Ng0=image.shape[0];Nge=target_Npix*osam
    dpix=((Ng0-Nge)/2);edx=(dx*osam);edy=(dy*osam)
    outI=np.zeros((target_Npix,target_Npix),dtype='float')
    #print(dpix,edx,edy)
    if space == 'real':
        for ic in range(Ng0):
            f1 = interpolate.interp1d(range(Ng0), image[ic] ,kind=kind)
            newx=np.arange(0,Nge,1)+dpix-edx
            newf1[ic][0:Nge]=f1(newx)
        for ic in range(Nge):
            f2 = interpolate.interp1d(range(Ng0), newf1[:,ic], kind=kind)
            newy=np.arange(0,Nge,1)+dpix-edy
            newf1[0:Nge,ic]=f2(newy)
        newf1 = newf1[0:Nge, 0:Nge]
        outI=newf1.reshape(target_Npix, osam, target_Npix, osam).mean(-1).mean(1)
    if space == 'log':
        sign = np.sign(image)
        log_star = np.log(np.fabs(image)+1.)*sign
        for ic in range(Ng0):
            f1 = interpolate.interp1d(range(Ng0), log_star[ic] ,kind=kind)
            newx=np.arange(0,Nge,1)+dpix-edx
            newf1[ic][0:Nge]=f1(newx)
        for ic in range(Nge):
            f2 = interpolate.interp1d(range(Ng0), newf1[:,ic], kind=kind)
            newy=np.arange(0,Nge,1)+dpix-edy
            newf1[0:Nge,ic]=f2(newy)
        newf1 = newf1[0:Nge, 0:Nge]
        sign = np.sign(newf1)
        outI = (np.exp(np.fabs(newf1))-1.)*sign
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


def interp_fft(image,target_Npix,dx,dy,
                 osam=1,
                 space='real'):
    # print(image.shape,target_Npix,dx,dy,osam)
    newf1=np.zeros(image.shape,dtype='float')
    Ng0=image.shape[0];Nge=target_Npix*osam
    dpix=int(np.round(((Ng0-Nge)/2)));edx=(dx*osam);edy=(dy*osam)
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



def polyfit4pix_vectorized(pos, val, tar_pos, polyorder):
    """
    PSF field interpolation modual
    pos: numpy.ndarray, observed PSF positions
    val: numpy.ndarray, PSF models at positions
    tar_pos: numpy.ndarray, interpolated PSF positions
    polyorder: int , polynomial order used in interpolation
    """
    Nstar, Ng, _ = val.shape
    iNstar = tar_pos.shape[0]

    def build_design_matrix(points):
        l0 = sum(range(polyorder + 2))
        X = np.ones((points.shape[0], l0))
        idx = 0
        for i in range(polyorder + 1):
            for j in range(polyorder + 1):
                if i + j <= polyorder:
                    X[:, idx] = points[:, 0]**i * points[:, 1]**j
                    idx += 1
        return X

    X = build_design_matrix(pos)       # (Nstar, l0)
    Xtx = X.T @ X                      # (l0, l0)

    val_flat = val.reshape(Nstar, -1)  # (Nstar, Ng*Ng)
    Xty_flat = X.T @ val_flat          # (l0, Ng*Ng)

    # Batch solve all linear systems at once
    try:
        coeffs_solved_flat = np.linalg.solve(Xtx, Xty_flat)  # (l0, Ng*Ng)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix encountered during linear solve.")

    coeffs = coeffs_solved_flat.reshape(Xtx.shape[0], Ng, Ng)  # (l0, Ng, Ng)

    Xi = build_design_matrix(tar_pos)  # (iNstar, l0)

    # Evaluate at target positions using einsum
    fitted = np.einsum('il,lmn->imn', Xi, coeffs)  # (iNstar, Ng, Ng)

    return fitted



#pulic declaretions
libdir = os.path.dirname(os.path.abspath(__file__))
_exts = ['.so', '.pyd', '.dll']
_fnlib = [f for f in os.listdir(libdir)
          if 'libpsffit' in f and any(f.endswith(e) for e in _exts)]
if not _fnlib:
    raise RuntimeError(
        f"Cannot find libpsffit shared library in {libdir}. "
        "Make sure the C extension was built successfully."
    )
fnlib = _fnlib[0]
psflibdir = os.path.join(libdir, fnlib)
libpsffit = ctypes.CDLL(psflibdir)
fpoint=ctypes.POINTER(ctypes.c_float)
dpoint=ctypes.POINTER(ctypes.c_double)
ipoint=ctypes.POINTER(ctypes.c_int)

def centriod_psf(psf):
    '''
    intput the psf arry with Ng,Ng,
    and return the estimated pixel center of the profile :cx,cy
    based on fast shape estimate algorithm 
    '''
    Ng1=psf.shape[0]
    Ng2=psf.shape[1];
    in_image=(Ng1*Ng2*ctypes.c_float)(*list(np.ravel(psf)))
    cent=(3*ctypes.c_double)()
    libpsffit.centriod_psflib.argtypes=[fpoint,ctypes.c_int,ctypes.c_int,dpoint]
    libpsffit.centriod_psflib(in_image,Ng1,Ng2,cent)
    return(cent[0],cent[1],cent[2])

