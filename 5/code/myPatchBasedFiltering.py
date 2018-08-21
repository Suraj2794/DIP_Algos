import numpy as np
from matplotlib import pyplot as plt
import hdf5storage
import math
import cv2


def patch_filtered_image(img_array,window_size,sigma_s):
	window=np.zeros(window_size,dtype=float)	#creating window array of user specified size
	window_center_r=int(window_size[0]/2) #calculating center of the window
	window_center_c=int(window_size[1]/2)
	filtered_image=np.zeros_like(img_array)
	(x,y)=img_array.shape
	for i in range(x):
		print(i)
		for j in range(y):
			window=np.zeros(window_size,dtype=float)  #getting window of user specified size with image pixels in it and placing image pixel at center
			#print(i,j,window_center_r,window_center_r+min(window_size[0]-window_center_r,x-i),window_center_c,window_center_c+min(window_size[0]-window_center_c,y-j))
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_array[i:(i+min(x-i,window_size[0]-window_center_r)),j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c-min(j,window_center_c):window_center_c+1]=img_array[i:(i+min(x-i,window_size[0]-window_center_r)),(j-min(j,window_center_c)):j+1]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_array[i-min(i,window_center_r):i+1,j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c-min(j,window_center_c):window_center_c+1]=img_array[i-min(i,window_center_r):i+1,(j-min(j,window_center_c)):j+1]
			#a=np.nonzero(window)
			#x_s=a[0][0]
			#y_s=a[1][0]
			#x_e=a[0][len(a[0])-1]
			#y_e=a[1][len(a[1])-1]
			#print(window)
			filtered=cal_inner_patch_val(window,(9,9),(window_center_r,window_center_c),sigma_s)
			sum_filt=0
			for k in range(filtered.shape[0]):
				for l in range(filtered.shape[1]):
					sum_filt+=filtered[k,l]*window[k,l]
			filtered_image[i,j]=sum_filt#np.linalg.norm((filtered[...,None]*window[:,None]).reshape(filtered.shape[0],-1),1)
	return filtered_image



def cal_inner_patch_val(img_array,window_size,center_window,sigma_s):
	base_window=np.zeros(window_size,dtype=float)   
	window_center_r=int(window_size[0]/2) 
	window_center_c=int(window_size[1]/2)
	(x,y)=img_array.shape
	(i,j)=center_window
	filtered_image=np.zeros_like(img_array)
	base_window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_array[i:(i+min(x-i,window_size[0]-window_center_r)),j:(j+min(window_size[1]-window_center_c,y-j))]
	base_window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c-min(j,window_center_c):window_center_c+1]=img_array[i:(i+min(x-i,window_size[0]-window_center_r)),(j-min(j,window_center_c)):j+1]
	base_window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_array[i-min(i,window_center_r):i+1,j:(j+min(window_size[1]-window_center_c,y-j))]
	base_window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c-min(j,window_center_c):window_center_c+1]=img_array[i-min(i,window_center_r):i+1,(j-min(j,window_center_c)):j+1]
	sum_window=0
	for i in range(x):
		for j in range(y):
			window=np.zeros(window_size,dtype=float)  #getting window of user specified size with image pixels in it and placing image pixel at center
			#print(i,j,window_center_r,window_center_r+min(window_size[0]-window_center_r,x-i),window_center_c,window_center_c+min(window_size[0]-window_center_c,y-j))
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_array[i:(i+min(x-i,window_size[0]-window_center_r)),j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c-min(j,window_center_c):window_center_c+1]=img_array[i:(i+min(x-i,window_size[0]-window_center_r)),(j-min(j,window_center_c)):j+1]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_array[i-min(i,window_center_r):i+1,j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c-min(j,window_center_c):window_center_c+1]=img_array[i-min(i,window_center_r):i+1,(j-min(j,window_center_c)):j+1]
			norm=cal_norm(window-base_window,sigma_s)
			sum_window+=norm
			filtered_image[i,j]=norm
	return filtered_image/sum_window


def cal_norm(diff_arr,sigma):
    return math.exp(-1*np.linalg.norm(diff_arr,2)**2/(sigma**2))



def corrupt_img_gray(img_arr,mean):
	img_res=np.zeros_like(img_arr)
	shape=img_arr.shape
	variance=0.05*(np.amax(img_arr)-np.amin(img_arr))
	gauss_noise=np.random.normal(mean,variance,shape)
	img_res=img_arr+gauss_noise
	return img_res


def cal_rms(org_arr,fil_arr):
	shape=org_arr.shape
	n=shape[0]*shape[1]
	sum_diff=0
	for i in range(shape[0]):
		for j in range(shape[1]):
			sum_diff+=(org_arr[i,j]-fil_arr[i,j])**2
	return math.sqrt(sum_diff/n)


if __name__ == '__main__':
	barba_mat = hdf5storage.loadmat('../data/barbara.mat')
	barba_array=np.array(barba_mat['imageOrig'])
	barba_array=barba_array/np.amax(barba_array)
	barba_noisy=corrupt_img_gray(barba_array,0)
	plt.imsave('../images/noisy_barbara.png',barba_noisy,cmap='gray')
	plt.imsave('../images/original_barbara.png',barba_array,cmap='gray')
	print(barba_noisy[::2][:,::2].shape)
	barba_blur=cv2.GaussianBlur(barba_noisy[::2][:,::2],(3,3),1)
	barba_trans=patch_filtered_image(barba_blur,(25,25),0.5)
	plt.imsave('../images/barba_filtered.png',barba_trans,cmap='gray')


	fig=plt.figure(figsize=(30,30))
	fig.add_subplot(1,3,1)
	plt.imshow(barba_array,cmap='gray')
	plt.colorbar()
	fig.add_subplot(1,3,2)
	plt.imshow(barba_noisy,cmap='gray')
	plt.colorbar()
	fig.add_subplot(1,3,3)
	plt.imshow(barba_trans,cmap='gray')
	plt.colorbar()
   ####################################################

	grass_array=plt.imread('../data/grass.png')
	grass_noisy=corrupt_img_gray(grass_array,0)
	plt.imsave('../images/noisy_grass.png',grass_noisy,cmap='gray')
	plt.imsave('../images/original_grass.png',grass_array,cmap='gray')
	grass_trans=patch_filtered_image(grass_noisy,(25,25),2)
	plt.imsave('../images/grass_filtered.png',grass_trans,cmap='gray')


	fig1=plt.figure(figsize=(30,30))
	fig1.add_subplot(1,3,1)
	plt.imshow(grass_array,cmap='gray')
	plt.colorbar()
	fig1.add_subplot(1,3,2)
	plt.imshow(grass_noisy,cmap='gray')
	plt.colorbar()
	fig1.add_subplot(1,3,3)
	plt.imshow(grass_trans,cmap='gray')
	plt.colorbar()

	#####################################

	honey_array=plt.imread('../data/honeyCombReal.png')
	honey_noisy=corrupt_img_gray(honey_array,0)
	plt.imsave('../images/noisy_honey.png',honey_noisy,cmap='gray')
	plt.imsave('../images/original_honey.png',honey_array,cmap='gray')
	honey_trans=patch_filtered_image(honey_noisy,(25,25),2)
	plt.imsave('../images/honey_filtered.png',honey_trans,cmap='gray')


	fig2=plt.figure(figsize=(30,30))
	fig2.add_subplot(1,3,1)
	plt.imshow(honey_array,cmap='gray')
	plt.colorbar()
	fig2.add_subplot(1,3,2)
	plt.imshow(honey_noisy,cmap='gray')
	plt.colorbar()
	fig2.add_subplot(1,3,3)
	plt.imshow(honey_trans,cmap='gray')
	plt.colorbar()


	print("barbara RMS:--- ",cal_rms(barba_array,barba_trans))
	print("grass RMS:---- ",cal_rms(grass_array,grass_trans))
	print("honey RMS:--- ",cal_rms(honey_array,honey_trans))

	plt.show()
