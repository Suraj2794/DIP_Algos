import numpy as np
from matplotlib import pyplot as plt
import hdf5storage
import math
from sklearn.neighbors import NearestNeighbors,RadiusNeighborsRegressor
import pickle as p
import cv2


def create_5_data(img_arr):   # create 3 dimensional data from pixels 
	r=img_arr.shape[0]*img_arr.shape[1]
	mean_data=np.zeros((img_arr.shape[0]*img_arr.shape[1],5))
	data=np.zeros((img_arr.shape[0]*img_arr.shape[1],5))
	num=0
	for i in range(img_arr.shape[0]):
		for j in range(img_arr.shape[1]):
			data[num,:3]=img_arr[i,j]
			data[num,3]=i
			data[num,4]=j
			num+=1
	return data

def fnc_gaussian(x,sigma):   # function to calculate gaussian 
	return np.exp(-(x ** 2) / (2 * sigma ** 2))

def cal_smooth(data,sigma_s,sigma_r,iter):  # function to segment image
	data_c=np.copy(data)
	for k in range(iter):  #iterate through each pixel
		print(k)
		nbrs = NearestNeighbors(n_neighbors=100,algorithm='auto').fit(data_c) # Using nearest neighbour approach 
		dis,ind=nbrs.kneighbors(data_c)
		for i in range(ind.shape[0]):
			nn=data_c[list(ind[i])]
			nn_s=nn[:,3:]-data_c[i,3:]
			nn_s_n=np.linalg.norm(nn_s,2,axis=1)
			w_s=fnc_gaussian(nn_s_n,sigma_s)  #calculating gaussian spatial kernel 
			nn_r=nn[:,:3]-data_c[i,:3]
			nn_r_n=np.linalg.norm(nn_r,2,axis=1)
			w_r=fnc_gaussian(nn_r_n,sigma_r)  # calculating gaussian range kernel
			mul=w_s*w_r
			a=(mul).reshape(ind[i].shape[0],1).T
			data_c[i,:3]=np.dot(a,nn[:,:3])/np.linalg.norm(mul,1)  
	return data_c

def create_img(data_c,img_arr):   # after segmentation create back image from 5d data
	img_op=np.zeros_like(img_arr[::2,::2])
	for d in data_c:
		img_op[int(d[3]),int(d[4])]=d[:3]
	return img_op


if __name__ == '__main__':
	baboon_array=plt.imread('../data/baboonColor.png')
	smooth_baboon=cv2.GaussianBlur(baboon_array,(3,3),1) # smoothing baboon image before downsampling
	data=create_5_data(smooth_baboon[::2,::2])  # create a 5d data
	data_c=cal_smooth(data,3,0.1,3)  
	babbon_img=create_img(data_c,baboon_array)

	plt.imsave('../images/baboon_orignal.png',baboon_array)
	plt.imsave('../images/baboon_smooth.png',smooth_baboon[::2,::2])
	plt.imsave('../images/baboon_mean_shift.png',babbon_img)

	fig=plt.figure(figsize=(30,30))
	fig.add_subplot(1,3,1)
	plt.imshow(smooth_baboon[::2,::2])
	plt.title("Smooth Baboon Image")
	plt.colorbar()
	fig.add_subplot(1,3,2)
	plt.imshow(babbon_img)
	plt.title("Mean Shift Image without clustering")
	plt.colorbar()


	#after calculating segmented image calculating clusters based on radius parameter across intensities values

	nbrs = NearestNeighbors(radius=0.007,algorithm='auto').fit(data_c[:,:3])
	dis,ind=nbrs.radius_neighbors(data_c[:,:3])

	new_data=np.copy(data_c)
	for i in range(len(ind)):
		nn=data_c[list(ind[i])]
		nn[:,:3]=new_data[i,:3]  # assign same pixel values with neighbours
		new_data[list(ind[i]),:3]=nn[:,:3]

	seg_img=np.zeros_like(baboon_array[::2,::2])

	for d in new_data:
		seg_img[int(d[3]),int(d[4])]=d[:3]
	

	plt.imsave('../images/baboon_clustered.png',seg_img)

	fig.add_subplot(1,3,3)
	plt.imshow(seg_img)
	plt.title("Mean Shift Image after clustering")
	plt.colorbar()
	plt.show()
