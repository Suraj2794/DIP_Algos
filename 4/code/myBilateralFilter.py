import numpy as np
from matplotlib import pyplot as plt
import hdf5storage
import math

def bilateral_filtered_image(img_array,window_size,sigma_r,sigma_s):
	window=np.zeros(window_size,dtype=float)	#creating window array of user specified size
	window_center_r=int(window_size[0]/2) #calculating center of the window
	window_center_c=int(window_size[1]/2)
	filtered_image=np.zeros_like(img_array)
	(x,y)=img_array.shape
	for i in range(x):
		for j in range(y):
			window=np.zeros(window_size,dtype=float)  #getting window of user specified size with image pixels in it and placing image pixel at center
			#print(i,j,window_center_r,window_center_r+min(window_size[0]-window_center_r,x-i),window_center_c,window_center_c+min(window_size[0]-window_center_c,y-j))
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_array[i:(i+min(x-i,window_size[0]-window_center_r)),j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c-min(j,window_center_c):window_center_c+1]=img_array[i:(i+min(x-i,window_size[0]-window_center_r)),(j-min(j,window_center_c)):j+1]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_array[i-min(i,window_center_r):i+1,j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c-min(j,window_center_c):window_center_c+1]=img_array[i-min(i,window_center_r):i+1,(j-min(j,window_center_c)):j+1]
			a=np.nonzero(window)
			x_s=a[0][0]
			y_s=a[1][0]
			x_e=a[0][len(a[0])-1]
			y_e=a[1][len(a[1])-1]
			filtered=calc_avrg(window,(x_s,y_s),(x_e,y_e),(window_center_r,window_center_c),sigma_r,sigma_s)
			filtered_image[i,j]=filtered
	return filtered_image


def calc_avrg(img_arr,start_pix,end_pix,cent_pix,sigma_r,sigma_s):
	filtered=0
	normal_weight=0
	for i in range(start_pix[0],end_pix[0]):
		for j in range(start_pix[1],end_pix[1]):
			gauss_int_differ=fnc_gaussian(img_arr[cent_pix]-img_arr[i,j],sigma_r)
			gauss_spat_differ=fnc_gaussian(pix_dist((i,j),(cent_pix)),sigma_s)
			mult=gauss_spat_differ*gauss_int_differ
			filtered+=img_arr[i,j]*mult
			normal_weight+=mult
	return (filtered/normal_weight)


def pix_dist(pix,cent_pix):
	return np.sqrt((pix[0]-cent_pix[0])**2+(pix[1]-cent_pix[1])**2)

def fnc_gaussian(x,sigma):
	return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))



def corrupt_img_gray(img_arr,mean):
	img_res=np.zeros_like(img_arr)
	shape=img_arr.shape
	variance=0.05*(np.amax(img_arr)-np.amin(img_arr))
	gauss_noise=np.random.normal(mean,variance,shape)
	img_res=img_arr+gauss_noise
	return img_res,gauss_noise


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
	#barba_array=barba_array/np.amax(barba_array)
	barba_noisy,gauss_noise=corrupt_img_gray(barba_array,0)
	plt.imsave('../images/noisy_barbara.png',barba_noisy,cmap='gray')
	plt.imsave('../images/original_barbara.png',barba_array,cmap='gray')
	barba_trans=bilateral_filtered_image(barba_noisy,(10,10),12,12)
	plt.imsave('../images/barba_filtered.png',barba_trans,cmap='gray')


	fig=plt.figure(figsize=(30,30))
	fig.add_subplot(2,2,1)
	plt.imshow(barba_array,cmap='gray')
	plt.colorbar()
	fig.add_subplot(2,2,2)
	plt.imshow(barba_noisy,cmap='gray')
	plt.colorbar()
	fig.add_subplot(2,2,3)
	plt.imshow(barba_trans,cmap='gray')
	plt.colorbar()
	fig.add_subplot(2,2,4)
	plt.imshow(gauss_noise,cmap='gray')
	plt.colorbar()

   ####################################################

	grass_array=plt.imread('../data/grass.png')
	grass_noisy,gauss_noise=corrupt_img_gray(grass_array,0)
	plt.imsave('../images/noisy_grass.png',grass_noisy,cmap='gray')
	plt.imsave('../images/original_grass.png',grass_array,cmap='gray')
	grass_trans=bilateral_filtered_image(grass_noisy,(3,3),5,5)
	plt.imsave('../images/grass_filtered.png',grass_trans,cmap='gray')


	fig1=plt.figure(figsize=(30,30))
	fig1.add_subplot(2,2,1)
	plt.imshow(grass_array,cmap='gray')
	plt.colorbar()
	fig1.add_subplot(2,2,2)
	plt.imshow(grass_noisy,cmap='gray')
	plt.colorbar()
	fig1.add_subplot(2,2,3)
	plt.imshow(grass_trans,cmap='gray')
	plt.colorbar()
	fig1.add_subplot(2,2,4)
	plt.imshow(gauss_noise,cmap='gray')
	plt.colorbar()

	#####################################

	honey_array=plt.imread('../data/honeyCombReal.png')
	honey_noisy,gauss_noise=corrupt_img_gray(honey_array,0)
	plt.imsave('../images/noisy_honey.png',honey_noisy,cmap='gray')
	plt.imsave('../images/original_honey.png',honey_array,cmap='gray')
	honey_trans=bilateral_filtered_image(honey_noisy,(3,3),5,5)
	plt.imsave('../images/honey_filtered.png',honey_trans,cmap='gray')


	fig2=plt.figure(figsize=(30,30))
	fig2.add_subplot(2,2,1)
	plt.imshow(honey_array,cmap='gray')
	plt.colorbar()
	fig2.add_subplot(2,2,2)
	plt.imshow(honey_noisy,cmap='gray')
	plt.colorbar()
	fig2.add_subplot(2,2,3)
	plt.imshow(honey_trans,cmap='gray')
	plt.colorbar()
	fig2.add_subplot(2,2,4)
	plt.imshow(gauss_noise,cmap='gray')
	plt.colorbar()


	print("barbara RMS:--- ",cal_rms(barba_array,barba_trans))
	print("grass RMS:---- ",cal_rms(grass_array,grass_trans))
	print("honey RMS:--- ",cal_rms(honey_array,honey_trans))

	plt.show()