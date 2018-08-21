import numpy as np
from PIL import Image  ##This library is used to show images (imp)
from matplotlib import pyplot as plt


def bin_interpol(img_arr,new_shape):   
	img_res=np.ndarray((new_shape[0],new_shape[1])) #create an array of given new shape
	r_ratio=0
	c_ratio=0
	if img_arr.shape[0]>new_shape[0]:
		r_ratio=img_arr.shape[0]/new_shape[0]  # based on new and old values calculating r_ratio
	else:
		r_ratio=(img_arr.shape[0]-1)/new_shape[0]  
	if img_arr.shape[1]<new_shape[1]:
		c_ratio=img_arr.shape[1]/new_shape[1]    # based on new and old values calculating c_ratio
	else:
		r_ratio=(img_arr.shape[1]-1)/new_shape[1]
	for i in range(new_shape[0]):
		for j in range(new_shape[1]):
			rf=i*r_ratio   #calculating pixel position to interpolate in old image
			cf=j*c_ratio
			r_diff=rf-int(rf)   
			c_diff=cf-int(cf)
			img_res[i,j]=img_arr[int(rf),int(cf)]*(1-r_diff)*(1-c_diff)+img_arr[min(int(rf)+1,img_arr.shape[0]-1),int(cf)]*(r_diff)*(1-c_diff)+img_arr[int(rf),min(int(cf)+1,img_arr.shape[1]-1)]*(1-r_diff)*(c_diff)+img_arr[min(int(rf)+1,img_arr.shape[0]-1),min(int(cf)+1,img_arr.shape[1]-1)]*r_diff*c_diff #formula for calculating binary interpolation
	return img_res

if __name__ == '__main__':
	img_barbara=Image.open("../data/barbaraSmall.png") # Open Image using PIL
	img_barbara_arr=np.array(img_barbara)  # converting image into numpy array
	(r,c)=img_barbara_arr.shape
	img_barbara_bi_res=bin_interpol(img_barbara_arr,(r*3-1,c*2-1))  #calculate new image of different size using function defined above
	img_barbara.show()
	Image.fromarray(img_barbara_bi_res.astype('uint8')).show()
	fig=plt.figure(figsize=(8,8))  #selecting plot size to show
	fig.add_subplot(2,1,1)
	plt.imshow(Image.fromarray(img_barbara_bi_res.astype('uint8')),cmap='gray')
	plt.colorbar()
	fig.add_subplot(2,1,2)
	plt.imshow(img_barbara,cmap='gray')
	plt.colorbar()
	plt.show()