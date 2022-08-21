import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio
from medpy.filter.smoothing import anisotropic_diffusion

#open img witout cv2
path=r'C:\Users\ladan\Desktop\clustring\breast.png'
img = mpimg.imread(path)
imgplot = plt.imshow(img)
plt.show()
#size
wid = img.shape[1]
hgt = img.shape[0]
# displaying the dimensions
print(str(wid))
print(str(hgt))
##CV2
img = cv2.imread(r'C:\Users\ladan\Desktop\clustring\breast.png')
gauss = np.random.normal(0,1,img.size)
gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
noise = img + img * gauss
cv2.imshow('a',noise)
cv2.waitKey(0)
imgplot = plt.imshow(noise)
plt.show()
plt.imsave(r'C:\Users\ladan\Desktop\clustring\converted noisy.png',noise)
#######################################################################################
#denoising with anisotropic_diffusion that is suitable for TV images
noisy_img= cv2.imread(r'C:\Users\ladan\Desktop\clustring\converted noisy.png')
img_cleaned = anisotropic_diffusion(noisy_img,niter=80,kappa=60,gamma=0.3,option=1)
imgplot = plt.imshow(img_cleaned)
plt.show()
###psn
psnr_noisy = peak_signal_noise_ratio(img,noisy_img)
psnr_cleaned = peak_signal_noise_ratio(img, img_cleaned)
print("psnr for noisy image is:",psnr_noisy)
print("psnr for cleand image is:",psnr_cleaned)
##########################################################################################
#mtching and 3d filtering
import bm3d
ref_img = cv2.imread(r'C:\Users\ladan\Desktop\clustring\breast.png')
noisy_img= cv2.imread(r'C:\Users\ladan\Desktop\clustring\converted noisy.png')
BM3D_denoise_image=bm3d.bm3d(noisy_img,sigma_psd=0.2,stage_arg=bm3d.BM3DStages.ALL_STAGES)
#usual plot
imgplot = plt.imshow(BM3D_denoise_image)
plt.show()
#plot color foe each band
imgplot = plt.imshow(BM3D_denoise_image[:,:,0])
plt.show()
#ploting one band(the refrence picture is RGB):in matlab:imshow(img.[0,255])
imgplot = plt.imshow(BM3D_denoise_image[:,:,0], cmap='gray', vmin = 0, vmax = 255)
plt.show()
#
psnr_cleaned_two= peak_signal_noise_ratio(ref_img, BM3D_denoise_image)
print("psnr for noisy image is:",psnr_noisy)
print("psnr for 3dcleand image is:",psnr_cleaned_two)
