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

import sys
import os

# 动态添加当前目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#define the initial model
from scipy.interpolate import interp2d  
from astropy.io import fits
import ctypes

from scipy.stats import sigmaclip
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from scipy.ndimage import zoom 
gspars = galsim.GSParams()

import torch
from time import time
from scipy.sparse import diags
from scipy.signal import fftconvolve,convolve2d
#from C_tools_lib import centriod_galaxy as centriod_psf
from .utils import pre_data, centriod_psf, interp_cubic_ex, periodic_extension, image2magnify
tdtype = torch.float32


t1 = time()



def RPCA(instars,inmask,inspos,npc=10,osam=2,regular=(1,1,1,1),
          center_smooth_radius=-1, smooth_strength=1., niter_max = 20,device='cpu'):
    if niter_max <= 0:
        raise ValueError("niter_max must be greater than 0")
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
    magPCs,weights,Shifts,Coeffs,slecstar,errors=pre_data(stars,inmask,spos,npc,osam,nbound=4)
    
    Nobj,Ng0,_ = stars.shape
    mask = mask.reshape((Nobj,Ng0*Ng0))
    #weights = mask
    
    kNg = 5
    kNh = kNg//2
    kernel = np.ones((kNg,kNg))
    x,y = np.arange(kNg),np.arange(kNg)
    X,Y = np.meshgrid(x,y)
    R = ((X-kNh)**2+(Y-kNh)**2)**0.5
    sigma = 1.5
    kernel = np.exp(-R**2/(sigma**2))
    indx = np.where(R>kNh)
    kernel[indx] *= 0
    kernel /= np.sum(kernel)
    
    
    print(slecstar.shape)
    cerrors = []
    for ic in range(len(errors)):
        cx, cy,_ = centriod_psf(stars[ic])
        dx, dy = (Ng0/2.-0.5) - cx, (Ng0/2.-0.5) - cy
        '''cerror = interp_cubic_ex((errors[ic]*10000), 
                                 dx=dy, 
                                 dy=dx, 
                                 target_Npix = Ng0, 
                                 osam=1, 
                                 space='log')/10000'''
        cerror = interp_cubic_ex((errors[ic]*10000), 
                                 dx=dy, 
                                 dy=dx, 
                                 target_Npix = Ng0, 
                                 osam=1, 
                                 space='log')/10000
        cerrors.append(cerror**2)
    cerrors = np.array(cerrors)
    msigma = np.mean(cerrors, axis = 0)
    ext_msigma = periodic_extension(msigma, pad_width = 5)
    msigma = fftconvolve(ext_msigma, kernel, mode='same')
    msigma = msigma[5:Ng0+5,5:Ng0+5]
    msigma = image2magnify(msigma, osam)
    mweight = (1./msigma)
    mweight /= np.min(mweight)
    #mweight -= 1.
    #write_mult_fits('/home/linn/data/test.fits', mweight)
    
    print('mweight.shape:', mweight.shape)
    #mweight = np.exp(image2magnify(np.log(mweight), osam))
    
    
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
    #for l in range(len(magPCs)):
    #    plt.imshow(magPCs[l].reshape(Ng0*osam,Ng0*osam));plt.title('magPC:%d'%(l+1));plt.show()
    
    coeffs=Coeffs.copy()
    stars=slecstar.copy()#restar.shape is not the same as stars
    restars=np.zeros(stars.shape,dtype='float')
    chi2=0
    PCs=magPCs.copy()

    #constructing the shift kernels
    print("constructing the downsizing and shift kernels")
    Down_size = create_downsampling_matrix_torch(Ng0*osam, osam, device=device)
    print("create_downsampling_matrix_torch down")
    S_matrix = []
    for ic in range(len(Shifts)):
        t1=time()
        # kernel = S_ma_torch(Ng0*osam, Ng0*osam, Shifts[ic,1]*osam, Shifts[ic,0]*osam, n=20*osam)
        kernel = S_ma_torch(Ng0*osam, Ng0*osam, Shifts[ic,1]*osam, Shifts[ic,0]*osam, device=device)
        S_matrix.append(Down_size @ kernel)
        t2=time()
        print(ic," time:",t2-t1)
    print("preparing linear matrix")
    S_matrix = torch.stack(S_matrix)
    
    S_matrix = torch.tensor(S_matrix, dtype=tdtype).to(device)
    stars = torch.tensor(stars, dtype=tdtype).to(device)
    PCs = torch.tensor(PCs, dtype=tdtype).to(device)
    weights = torch.tensor(weights, dtype=tdtype).to(device)
    restars = torch.zeros_like(stars, dtype=tdtype).to(device)
    Coeffs = torch.tensor(coeffs, dtype=tdtype).to(device)
    
    
    ma_list = pre_matrix3_torch(S_matrix, 
                               Down_size,
                               stars,
                               weights,
                               mweight,
                               *regular,
                               center_smooth_radius, 
                                smooth_strength,
                                device=device)
    
    
    print("starting iterations")
    while niter<niter_max:
        print("PCs.shape:",PCs.shape)
        Coeffs = Solving_Coeff1(stars,weights,PCs,osam,S_matrix, device=device)
        PCs = Solving_PCs1(stars,weights,S_matrix,PCs,osam,Coeffs,Down_size,regular,ma_list, device=device)
        restars[:,:] *= 0.
        count = 0
        chi2 = 0
        for ic in range(Nobj):
            for l in range(npc):
                SPC = (S_matrix[ic] @ PCs[l])
                restars[ic] += SPC*Coeffs[ic][l]

            chi2 += torch.sum(((stars[ic]-restars[ic])**2)*weights[ic])
        detchi2=torch.abs(init_chi/chi2-1.)
        init_chi=chi2
        niter+=1
        print('niter=%d,detchi2=%f,chi2=%f'%(niter,detchi2,chi2/Nobj/Ng0/Ng0))
        
        judge_chi=init_chi/Nobj/Ng0/Ng0
        if detchi2<1.0e-8:
            break
            
    #calculate the variance of coefficients
    weights = weights.cpu().numpy()
    #Coeffs = Coeffs.cpu().numpy()
    stars = stars.cpu().numpy()
    Coeffs = []
    for ic in range(Nobj):
        shifPC=[]
        for l in range(npc):
            cPC = S_matrix[ic] @ PCs[l]
            cPC = cPC.cpu().numpy()
            shifPC.append(cPC)
        shifPC=np.array(shifPC)
        PtV=np.zeros(shifPC.shape,dtype='float')
        for l in range(npc):
            PtV[l,:] = shifPC[l,:]*weights[ic]
        PtVP = PtV @ shifPC.T
        tmp = PtV @ stars[ic]
        #iPtVP = np.linalg.inv(PtVP)
        #Coeff = iPtVP @ tmp
        #coeff = np.linalg.solve(PtVP,tmp)
        try:
            coeff = np.linalg.solve(PtVP, tmp)
        except:
            # singlar matrix
            print('singlar matrix')
            coeff = np.linalg.lstsq(PtVP, tmp).solution
        
        Coeffs.append(coeff)
    Coeffs=np.array(Coeffs);print('Solving_Coeff,Coeffs.shape',Coeffs.shape)
    # Coeffs = Coeffs.cpu().numpy()
    recoeffs=[]
    residual = []
    restars = np.zeros((Nobj,Ng0*Ng0))
    crestars = np.zeros((Nobj,Ng0*osam,Ng0*osam))
    for ic in range(Nobj):
        tmp=[]
        for l in range(npc):
            SPC = S_matrix[ic] @ PCs[l]
            SPC = SPC.cpu().numpy()
            Coeff_err=1./np.sum(SPC**2*weights[ic])
            tmp.append([Coeffs[ic][l],Coeff_err])
            restars[ic] += SPC*Coeffs[ic][l]
            crestars[ic] += (PCs[l].cpu().numpy() * Coeffs[ic][l]).reshape(Ng0*osam, Ng0*osam)
        recoeffs.append(tmp)
        residual.append(stars[ic]-restars[ic])
    recoeffs=np.array(recoeffs)
    residual = np.array(residual)
    
    return(PCs.cpu().numpy(),recoeffs,magPCs,coeffs,residual, crestars)


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


def Solving_Coeff1(stars,weights,PCs,osam,S_ma, device='cpu'):
    Nobj=stars.size()[0]
    npc=PCs.size()[0]
    Mp=stars.size()[1]
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
        shifPC = torch.stack(shifPC)
        PtV = torch.zeros_like(shifPC).to(device)
        for l in range(npc):
            PtV[l,:] = shifPC[l,:] * weights[ic,:]
        PtVP = PtV @ shifPC.t() 
        tmp = PtV @ stars[ic]
        #iPtVP = torch.linalg.inv(PtVP)
        #Coeff = iPtVP @ tmp
        #coeff = torch.linalg.solve(PtVP,tmp)
        # changed to deal with singlar condition
        try:
            coeff = torch.linalg.solve(PtVP, tmp)
        except torch._C._LinAlgError:
            # singlar matrix
            print('singlar matrix')
            coeff = torch.linalg.lstsq(PtVP, tmp).solution
        #print(shifPC.shape)
        #coeff = solve_with_normalization_and_error(shifPC.T, stars[ic], weights[ic])
        #sums = torch.sum(PCs.T @ coeff)
        #coeff /= sums
        #print("normalised flux is %.5f, star flux is %.5f"%(torch.sum(PCs.T @ coeff),torch.sum(stars[ic])))
        Coeffs.append(coeff)
    Coeffs=torch.stack(Coeffs);print('Solving_Coeff,Coeffs.shape',Coeffs.size())
    
    return(Coeffs)



def Solving_PCs1(stars,weights,S_matrix,PCs,osam,Coeffs,Down_size,regular, ma_list, device='cpu'):
    Nobj=stars.size()[0]
    Mp=stars.size()[1]
    Ng0 = int(Mp**0.5)
    npc=PCs.size()[0]
    newPCs=[]
    print('Training PCs')
    substars=stars.clone()
    #for l in range(npc):
    l=0
    while l < npc:
        PC = train_PC1(substars,weights,S_matrix,PCs[l],osam,Coeffs[:,l],Down_size,regular, ma_list, device=device)
        newPCs.append(PC)
        for i in range(Nobj):
            PCi = S_matrix[i] @ PC  # Move to CPU for numpy conversion
            substars[i] -= PCi * Coeffs[i,l]
        l+=1
    newPCs = torch.stack(newPCs).to(device)
    #orthogonalization
    for l in range(npc-1):            
        for ll in range(l + 1, npc):
            projection = torch.sum(newPCs[l]*newPCs[ll])
            newPCs[ll] -= projection*newPCs[l]
        sums=(torch.sum(newPCs[l]**2))**0.5 #normalization
        newPCs[l]/=sums
    #the last one
    sums = (torch.sum(newPCs[npc-1]**2))**0.5 #normalization
    newPCs[npc-1] /= sums
    return(newPCs)

def train_PC1(instars, weights, S_matrix, PC, osam, Coeff, Down_size, regular, ma_list, device='cpu'):
    stars = instars.clone()
    Nobj = stars.size()[0]
    num_epochs = 10
    A_torch, b_torch = constructAB3_torch(S_matrix, 
                                    Down_size, 
                                    stars, 
                                    weights, 
                                    PC,
                                    Coeff,
                                    ma_list,
                                    *regular,
                                    device=device)
    P_next = torch.linalg.solve(A_torch, b_torch)
    P_next /= (torch.sum(P_next**2))**0.5
    

    return (P_next) 




def Lanczos_torch(Ng, dx, dy, n=6, device='cpu'):
    '''
    PyTorch 版 Lanczos 核生成函数
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
    M = torch.arange(Ng, device=device, dtype=tdtype).to(device)
    y, x = torch.meshgrid(M, M, indexing='ij')  # PyTorch 默认 ij 索引，需转 xy
    
    # 坐标中心化并添加偏移量
    center = Ng / 2.0
    x = x - center - dx
    y = y - center - dy
    
    # 计算 Lanczos 函数
    x_pi = torch.pi * x
    y_pi = torch.pi * y
    
    # 计算 sinc 项 (注意 PyTorch 的 sinc 定义与 NumPy 一致: sin(πx)/(πx))
    sinc_x = torch.sinc(x_pi)  # 等价于 sin(πx)/(πx)
    sinc_x_window = torch.sinc(x_pi / n)  # 窗口函数
    
    sinc_y = torch.sinc(y_pi)
    sinc_y_window = torch.sinc(y_pi / n)
    
    # 组合 Lanczos 核
    lanczos = (sinc_x * sinc_x_window) * (sinc_y * sinc_y_window)
    
    # 应用截断条件 (|x| > n 或 |y| > n 的位置置零)
    mask = (x.abs() <= n) & (y.abs() <= n)
    lanczos = lanczos * mask.float()
    
    # 归一化
    lanczos_sum = lanczos.sum()
    if lanczos_sum != 0:  # 避免除以零
        lanczos /= lanczos_sum
    
    return lanczos


def create_downsampling_matrix_torch(Ng, osam, device='cpu'):
    '''
    Create a downsampling matrix using PyTorch tensors.
    '''
    ng = Ng // osam  # 计算降采样后的尺寸
    
    rows = []  # 存储每一行，即每个降采样核矩阵b的拉平版本
    
    for i in range(ng):
        for j in range(ng):
            b = torch.zeros((Ng, Ng), dtype=tdtype).to(device)  # 创建一个全零的降采样核矩阵b
            for di in range(osam):
                for dj in range(osam):
                    if (i * osam + di) < Ng and (j * osam + dj) < Ng:  # 确保索引不越界
                        b[i * osam + di, j * osam + dj] = 1  # 设置对应位置为1，表示累加
            rows.append(b.flatten())  # 将降采样核矩阵b拉平并添加到列表中

    B = torch.stack(rows).to(device)  # 将所有的行组合成一个大的矩阵B
    return B




def S_ma_torch(Ngi, Ng, dx, dy, n=10, device='cpu'):
    '''
    使用 PyTorch 实现带有周期性核的移位矩阵
    '''
    # 生成 Lanczos 核 (假设 Lanczos 函数已支持 PyTorch 张量)
    # kernel2d = Lanczos_torch(Ng, dx, dy, 20).to(device)  # [Ng, Ng]
    kernel2d = fft_kernel_torch(Ng, dx, dy, device=device).to(device) 
    ngi = Ngi
    ngp = Ng
    nhalf = ngp // 2

    # 生成索引网格 (PyTorch 实现)
    i, j = torch.meshgrid(
        torch.arange(ngi, device=device),
        torch.arange(ngi, device=device),
        indexing='ij'
    )  # 保持与 NumPy 相同的索引顺序
    
    it, jt = torch.meshgrid(
        torch.arange(ngi, device=device),
        torch.arange(ngi, device=device),
        indexing='ij'
    )

    # 计算相对位移 (通过广播自动扩展维度)
    x = i.unsqueeze(2).unsqueeze(3) - it.unsqueeze(0).unsqueeze(0)  # [ngi, ngi, 1, 1] - [1, 1, ngi, ngi]
    y = j.unsqueeze(2).unsqueeze(3) - jt.unsqueeze(0).unsqueeze(0)

    # 应用周期性边界条件 (PyTorch 的取模运算)
    is_mod = (x + nhalf) % ngp  # 自动处理负索引
    js_mod = (y + nhalf) % ngp

    # 收集核值并重塑矩阵
    tmp = kernel2d[is_mod.long(), js_mod.long()]  # 需要显式转换为 long 类型索引
    return tmp.view(ngi*ngi, ngi*ngi)  # 展平为 2D 矩阵


def fft_kernel_torch(Ng, dx, dy, device='cpu'):
    '''
    使用 PyTorch 实现基于 FFT 的位移核生成
    Args:
        device: 指定计算设备 ('cpu' 或 'cuda')
    '''
    # 生成坐标网格 (PyTorch 实现)
    M = torch.arange(Ng, device=device)
    x, y = torch.meshgrid(M, M, indexing='ij')
    x = x - Ng//2  # 中心化坐标
    y = y - Ng//2

    # 创建全1矩阵并计算 FFT
    a = torch.ones((Ng, Ng), dtype=tdtype, device=device)
    F_a = torch.fft.fft2(a)

    # 生成频率网格 (与 NumPy 的 fftfreq 等效)
    freq = torch.fft.fftfreq(Ng, device=device)
    kx, ky = torch.meshgrid(freq, freq, indexing='ij')  # [Ng, Ng]

    # 构建位移相位因子
    shift = F_a*0.
    shift.real += 1.
    phase = 2j * torch.pi * (kx * (-dy) + ky * (-dx))
    F_shift = shift * torch.exp(phase)  # 位移频域响应

    # 逆变换获取核函数
    kernel = torch.fft.ifft2(F_shift).real
    kernel = torch.fft.fftshift(kernel)  # 与 NumPy 的 fftshift 等效


    # 归一化处理
    kernel /= kernel.sum()

    return kernel


def create_dx(size, rows, cols, device='cpu'):
    Dx = torch.zeros(size, size).to(device)
    Dx += torch.diag(torch.full((size,), -1.0), diagonal=0).to(device)
    Dx += torch.diag(torch.full((size-1,), 1.0), diagonal=1).to(device)
    
    for i in range(1, rows):
        Dx[i * cols - 1, i * cols] = 0
    return Dx

def create_dy(size, rows, cols, device='cpu'):
    Dy = torch.zeros(size, size).to(device)
    Dy += torch.diag(torch.full((size,), -1.0), diagonal=0).to(device)
    Dy += torch.diag(torch.full((size-cols,), 1.0), diagonal=cols).to(device)
    return Dy

def create_d_d1(size, rows, cols, device='cpu'):
    Dd1 = torch.zeros(size, size).to(device)
    Dd1 += torch.diag(torch.full((size,), -1.0), diagonal=0).to(device)
    Dd1 += torch.diag(torch.full((size-cols-1,), 1.0), diagonal=cols+1).to(device)
    
    for i in range(1, rows):
        Dd1[i * cols - 1, i * cols] = 0
    return Dd1

def create_d_d2(size, rows, cols, device='cpu'):
    Dd2 = torch.zeros(size, size).to(device)
    Dd2 += torch.diag(torch.full((size,), -1.0), diagonal=0).to(device)
    Dd2 += torch.diag(torch.full((size-cols+1,), 1.0), diagonal=cols-1).to(device)
    
    for i in range(1, rows):
        Dd2[i * cols, i * cols - 1] = 0
    return Dd2

def first_order_diff_matrix_torch(n, shape, device='cpu'):
    
    # 参数设置
    rows, cols = shape
    size = rows * cols 

    # 创建各个方向的梯度矩阵
    Dx = create_dx(size, rows, cols, device=device)
    Dy = create_dy(size, rows, cols, device=device)
    Dd1 = create_d_d1(size, rows, cols, device=device)
    Dd2 = create_d_d2(size, rows, cols, device=device)
    return torch.vstack([Dx, Dy, Dd1, Dd2]).to(device)


def create_dxx(size, rows, cols):
    Dxx = torch.zeros(size, size)
    
    # 设置主对角线及其偏移为1和2的对角线
    Dxx += torch.diag(torch.full((size,), -2.0), diagonal=0)
    Dxx += torch.diag(torch.full((size-1,), 1.0), diagonal=1)
    Dxx += torch.diag(torch.full((size-2,), 1.0), diagonal=2)
    
    # 根据给定逻辑修改特定位置的值
    for i in range(1, rows):
        Dxx[i * cols - 2, i * cols] = 0  
        Dxx[i * cols - 1, i * cols + 1] = 0  
    return Dxx

def create_dyy(size, rows, cols):
    Dy = torch.zeros(size, size)
    
    # 设置主对角线及其偏移为列数和两倍列数的对角线
    Dy += torch.diag(torch.full((size,), -2.0), diagonal=0)
    Dy += torch.diag(torch.full((size-cols,), 1.0), diagonal=cols)
    Dy += torch.diag(torch.full((size-2*cols,), 1.0), diagonal=2*cols)
    
    return Dy

def create_dd1d1(size, rows, cols):
    Dd1 = torch.zeros(size, size)
    
    # 设置主对角线及其偏移为(cols+1)和2*(cols+1)的对角线
    Dd1 += torch.diag(torch.full((size,), -2.0), diagonal=0)
    Dd1 += torch.diag(torch.full((size-cols-1,), 1.0), diagonal=cols+1)
    Dd1 += torch.diag(torch.full((size-2*(cols+1),), 1.0), diagonal=2*(cols+1))
    
    # 根据给定逻辑修改特定位置的值
    for i in range(1, rows):
        Dd1[i * cols - 2, i * cols] = 0  
        Dd1[i * cols - 1, i * cols + 1] = 0  
    return Dd1

def create_dd2d2(size, rows, cols):
    Dd2 = torch.zeros(size, size)
    
    # 设置主对角线及其偏移为(cols-1)和2*(cols-1)的对角线
    Dd2 += torch.diag(torch.full((size,), -2.0), diagonal=0)
    Dd2 += torch.diag(torch.full((size-cols+1,), 1.0), diagonal=cols-1)
    Dd2 += torch.diag(torch.full((size-2*(cols-1),), 1.0), diagonal=2*(cols-1))
    
    # 根据给定逻辑修改特定位置的值
    for i in range(1, rows):
        Dd2[i * cols - 2, i * cols] = 0  
        Dd2[i * cols - 1, i * cols + 1] = 0  
    return Dd2


def second_order_diff_matrix_torch(n, shape, device='cpu'):
    rows, cols = shape
    size = rows * cols  

    # 创建各个方向的二阶差分矩阵
    Dxx = create_dxx(size, rows, cols)
    Dyy = create_dyy(size, rows, cols)
    Dd1d1 = create_dd1d1(size, rows, cols)
    Dd2d2 = create_dd2d2(size, rows, cols)

    # 组合多个方向的二阶差分
    D_second = torch.vstack([Dxx, Dyy, Dd1d1, Dd2d2])
    
    return D_second.to(device)



def constructAB3_torch(DS, D, I, sigma, P_init, Coeff, 
                       ma_tuple,
                       lambda_L2=1e-3, lambda_L1=0, lambda_TV=0, lambda_grad=0, device='cpu'):
    
    W_DS = ma_tuple[0]
    W_DS_DS = ma_tuple[1]
    W_DS_I = ma_tuple[2]
    ma_L2 = ma_tuple[3]
    ma_grad = ma_tuple[4]
    ma_TV = ma_tuple[5]
    DTD = ma_tuple[6]

    num_obs, Mp, Mpe = DS.size()
    #print("num_obs, Mp, Mpe:",num_obs, Mp, Mpe)
    A = torch.zeros((Mpe, Mpe), dtype=tdtype).to(device)
    b = torch.zeros(Mpe, dtype=tdtype).to(device)
    
    #print("DS.shape, D.shape, I.shape, sigma.shape, P_init.shape:",
    #     DS.shape, D.shape, I.shape, sigma.shape, P_init.shape)
    
    #print("W_DS.shape,W_DS_DS.shape,W_DS_I.shape,DTD.shape:",
    #     W_DS.shape,W_DS_DS.shape,W_DS_I.shape,DTD.shape)
    
    Coeff_squared = Coeff**2

    for i in range(num_obs):
        I_i = I[i] #.flatten()  # 观测数据 (Mp,)

        # 直接利用逐元素乘法避免生成对角矩阵
        #A += W_DS_DS[i] * Coeff[i]**2  # (Mpe, Mpe)
        #b += W_DS_I[i] * Coeff[i] - (W_DS_DS[i] @ P_init) * Coeff[i]**2
        A.add_(W_DS_DS[i] * Coeff_squared[i])
        b.add_(W_DS_I[i] * Coeff[i])

    # **1. L2 正则化（Tikhonov 正则化）**
    if lambda_L2 > 0:
        A += ma_L2  # 控制 ΔP 大小

    # **2. L1 正则化（稀疏性）**
    if lambda_L1 > 0:
        diag_L1 = torch.abs(P_init.flatten())  # 绝对值稀疏正则
        A += lambda_L1 * torch.diag(diag_L1)

    # **3. Total Variation (TV) 正则化**
    if lambda_TV > 0:
        # L = second_order_diff_matrix_torch(Mpe, (int(np.sqrt(Mpe)), int(np.sqrt(Mpe))))
        A += ma_TV  #二阶差分矩阵

    # **4. 一阶差分梯度正则化**
    if lambda_grad > 0:
        # D_grad = first_order_diff_matrix_torch(Mpe, (int(np.sqrt(Mpe)), int(np.sqrt(Mpe))))
        A += ma_grad #一阶差分矩阵
    
    # **5. 先验项**
    # A += DTD

    return A, b


# →、←、↑、↓。这些符号分别表示向右、向左、向上、向下的方向。对角箭头：↖、↗、↘、↙



def pre_matrix3_torch(DS, D, I, weight,mweight,
                      lambda_L2=1e-3, lambda_L1=0, lambda_TV=0, lambda_grad=0,
                      center_smooth_radius=-1, smooth_strength=1., device='cpu'):
    num_obs, Mp, Mpe = DS.shape

    # cal W_DS, W_DS_DS, W_DS_I
    W_DS = DS.permute(0, 2, 1) * weight[:, None, :]  # (num_obs, Mpe, Mp)
    W_DS_DS = torch.bmm(W_DS, DS)  # (num_obs, Mpe, Mpe)
    W_DS_I = torch.bmm(W_DS, I.unsqueeze(-1)).squeeze(-1)  # (num_obs, Mpe)
    osam = (Mpe/Mp)**0.5

    # cal DTD
    DTD = D.t() @ D

    # initialzing regularization term
    ma_L2 = 0
    ma_TV = 0
    ma_grad = 0
    
    
    # obtaining the image size
    height, width = int(np.sqrt(Mpe)), int(np.sqrt(Mpe))
    dpix = 0
    if osam%2 == 0:
        dpix = 0.5
    print('dpix=',dpix)
    if center_smooth_radius > 0:
        # constructing inverse distance weight matrix
        print('circular weight')
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        center_y, center_x = height // 2 - dpix, width // 2 - dpix
        distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        distance_weight = torch.clamp(distance - center_smooth_radius, 
                                      min=0)**smooth_strength
    else:
        print('noise weight')
        distance_weight = torch.tensor(mweight**smooth_strength, dtype=tdtype).to(device)
    

    distance_weight = distance_weight.flatten().to(DS.device) 

    # L2 regularization
    if lambda_L2 > 0:
        ma_L2 = lambda_L2 * torch.eye(Mpe, dtype=torch.float32, device=DS.device)

    # TV regularization（appling distance_weight）
    if lambda_TV > 0:
        # constructing diferential matrix
        height, width = int(np.sqrt(Mpe)), int(np.sqrt(Mpe))
        L = second_order_diff_matrix_torch(Mpe, (height, width), device=device)
        #weighted_L = L 
        weighted_L = L * distance_weight[None, :]  # 对行进行加权
        ma_TV = lambda_TV * (weighted_L.t() @ weighted_L) 

    # gradient regularization（appling distance_weight）
    if lambda_grad > 0:

        # constructing diferential matrix
        D_grad = first_order_diff_matrix_torch(Mpe, (height, width), device=device)
        #weighted_D_grad = D_grad 
        weighted_D_grad = D_grad * distance_weight[None, :]  # 对行进行加权
        ma_grad = lambda_grad * (weighted_D_grad.t() @ weighted_D_grad)

    return (
        W_DS,
        W_DS_DS,
        W_DS_I,
        ma_L2,
        ma_grad,
        ma_TV,
        DTD
    )






