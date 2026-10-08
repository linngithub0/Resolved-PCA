# Resolved-PCA
a Resolved PSF reconstruction method for undersampled images


The associated data and notebook scripts could be found at: https://nadc.china-vo.org/res/r101915/

Here are the Python code usage example:


from rpca.RPCA import RPCA


factor_L2 = (1/0.01)**2
factor_L1 = (1/0.01)**2
factor_Lap = (1/0.01)**2
factor_grad = (1/0.01)**2
regular = (factor_L2,factor_L1,factor_Lap,factor_grad)


RPCs,Coeffs,magPCs,coeffs,residual,rstars=RPCA(train_stars, # star stamps with shape (Nstar, Npix, Npix)
                                               train_mask,  # mask stamps with shape (Nstar, Npix, Npix), in which the invalid pixels are masked as 0, and the valid pixels are masked as 1.
                                               npc=20, # number of PCs to be calculated
                                               osam=osam, # oversampling factor, integer, 2 is recommended
                                               regular=regular, # regularization parameters
                                               smooth_strength=1.5, # radial smoothness constraints 
                                               center_smooth_radius = 2., # The radial smoothing parameter shielding range of the central pixel
                                               niter_max = 40, # rpca iteration times, 
                                               device='cuda:0')
