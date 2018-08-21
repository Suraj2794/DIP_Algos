import numpy as np
from matplotlib import pyplot as plt
import hdf5storage
import math
import cv2



def unsharp_image(img_arr,amount,sigma,kernel_window_size):
	blur_img=cv2.GaussianBlur(img_arr,kernel_window_size,sigma)
	sharp_img=img_arr+(img_arr-blur_img)*amount
	return sharp_img








if __name__ == '__main__':
	moon_mat=hdf5storage.loadmat('../data/superMoonCrop.mat')
	lion_mat=hdf5storage.loadmat('../data/lionCrop.mat')
	moon_array=np.array(moon_mat['imageOrig'])
	lion_array=np.array(lion_mat['imageOrig'])
	plt.imsave('../images/original_lion.png',lion_array,cmap='gray')
	plt.imsave('../images/original_moon.png',moon_array,cmap='gray')
	lion_res=unsharp_image(lion_array,5,1,(3,3))
	moon_res=unsharp_image(moon_array,15,1,(3,3))
	
	plt.imsave('../images/sharped_lion.png',lion_res,cmap='gray')
	plt.imsave('../images/sharped_moon.png',moon_res,cmap='gray')

	fig=plt.figure(figsize=(30,30))
	fig.add_subplot(1,2,1)
	plt.imshow(lion_array,cmap='gray')
	plt.colorbar()
	fig.add_subplot(1,2,2)
	plt.imshow(lion_res,cmap='gray')
	plt.colorbar()

	fig1=plt.figure(figsize=(30,30))
	fig1.add_subplot(1,2,1)
	plt.imshow(moon_array,cmap='gray')
	plt.colorbar()
	fig1.add_subplot(1,2,2)
	plt.imshow(moon_res,cmap='gray')
	plt.colorbar()

	plt.show()

