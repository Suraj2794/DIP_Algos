import numpy as np
from PIL import Image   ##This library is used to show images (imp) and load images
from matplotlib import pyplot as plt

def near_neighbour(img_arr,new_shape):
	img_res=np.ndarray((new_shape[0],new_shape[1]))  #create an array of given new shape
	r_ratio=0
	c_ratio=0
	if img_arr.shape[0]>new_shape[0]:
		r_ratio=img_arr.shape[0]/new_shape[0]   # based on new and old values calculating r_ratio
	else:
		r_ratio=(img_arr.shape[0]-1)/new_shape[0]
	if img_arr.shape[1]<new_shape[1]:
		c_ratio=img_arr.shape[1]/new_shape[1]   # based on new and old values calculating c_ratio
	else:
		r_ratio=(img_arr.shape[1]-1)/new_shape[1]
	print(r_ratio,c_ratio)
	for i in range(new_shape[0]):
		for j in range(new_shape[1]):
			rf=r_ratio*i       #calculating pixel position to interpolate in old image
			cf=c_ratio*j
			img_res[i,j]=img_arr[int(rf),int(cf)] #copy that particular pixel position into new image pixel position 
	return img_res


if __name__ == '__main__':
	img_barbara=Image.open("../data/barbaraSmall.png") ## Open Image using PIL
	img_barbara_arr=np.array(img_barbara)
	(r,c)=img_barbara_arr.shape
	img_barbara_nn_res=near_neighbour(img_barbara_arr,(r*3-1,c*2-1)) #calculate new image of different size using function defined above
	img_barbara.show()
	Image.fromarray(img_barbara_nn_res.astype('uint8')).show()
	fig=plt.figure(figsize=(8,8))
	fig.add_subplot(2,1,1)
	plt.imshow(Image.fromarray(img_barbara_nn_res.astype('uint8')),cmap='gray')
	plt.colorbar()
	fig.add_subplot(2,1,2)
	plt.imshow(img_barbara,cmap='gray')  #plot colormap 
	plt.colorbar()
	plt.show()