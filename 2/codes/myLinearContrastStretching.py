import numpy as np
from PIL import Image  ##This library is used to show and load images (imp)
from matplotlib import pyplot as plt

def lin_cont_str(img_arr):
	min_int=np.amin(img_arr)  # get max intensity in image
	max_int=np.amax(img_arr)  # get min intensity in image
	img_res=np.copy(img_arr)
	for i in range(img_arr.shape[0]):
		for j in range(img_arr.shape[1]):
			img_res[i,j]=min(((img_arr[i,j]-min_int)/(max_int-min_int))*255+min_int,255)  #formula for calculating linear contrast stretching
	return img_res


def lin_cont_str_col(img_arr): # linear contrast streching for colored images
	min_int=[]  # array for storing max values of all the three channels
	max_int=[]	# array for storing min values of all the three channels
	for k in range(img_arr.shape[2]):
		min_int.append(np.amin(img_arr[:,:,k]))  
		max_int.append(np.amax(img_arr[:,:,k]))
	img_res=np.copy(img_arr)
	for k in range(img_arr.shape[2]):
		for i in range(img_arr.shape[0]):
			for j in range(img_arr.shape[1]):
					img_res[i,j,k]=min(((img_arr[i,j,k]-min_int[k])/(max_int[k]-min_int[k]))*255,255) # formula for calculating linear contrast streching
	return img_res

if __name__ == '__main__':
	img_barba=Image.open("../data/barbara.png")   #Opening all given images using PIL 
	img_tem=Image.open("../data/TEM.png")
	img_canyon=Image.open("../data/canyon.png")
	img_church=Image.open("../data/church.png")
	img_retina=Image.open("../data/retina.png")
	img_tem_arr=np.array(img_tem)  #converting all PIL images into numpy nd array
	img_canyon_arr=np.array(img_canyon)
	img_barba_arr=np.array(img_barba)
	img_church_arr=np.array(img_church)
	img_retina_arr=np.array(img_retina)
	if(len(img_barba_arr.shape)) ==2:  # checking for colored images
		img_barba_res=lin_cont_str(img_barba_arr)
	elif (len(img_barba_arr.shape)) == 3:
		img_barba_res=lin_cont_str_col(img_barba_arr)
	
	img_barba.show()
	Image.fromarray(img_barba_res.astype('uint8')).show()

	if(len(img_tem_arr.shape)) ==2:  # checking if the image is colored or not
		img_tem_res=lin_cont_str(img_tem_arr)  #getting linear contrasted image
	elif (len(img_tem_arr.shape)) == 3:
		img_tem_res=lin_cont_str_col(img_tem_arr)

	img_tem.show()   # showing original image
	Image.fromarray(img_tem_res.astype('uint8')).show()  #showing image after applying linear contrast stretching

	## Same for all the below images

	if(len(img_canyon_arr.shape)) ==2: 
		img_canyon_res=lin_cont_str(img_canyon_arr)
	elif (len(img_canyon_arr.shape)) == 3:
		img_canyon_res=lin_cont_str_col(img_canyon_arr)

	img_canyon.show()
	Image.fromarray(img_canyon_res.astype('uint8')).show()

	if(len(img_church_arr.shape)) ==2:
		img_church_res=lin_cont_str(img_church_arr)
	elif (len(img_church_arr.shape)) == 3:
		img_church_res=lin_cont_str_col(img_church_arr)

	img_church.show()
	Image.fromarray(img_church_res.astype('uint8')).show()

	if(len(img_retina_arr.shape)) ==2:
		img_retina_res=lin_cont_str(img_retina_arr)
	elif (len(img_retina_arr.shape)) == 3:
		img_retina_res=lin_cont_str_col(img_retina_arr)

	img_retina.show()
	Image.fromarray(img_retina_res.astype('uint8')).show()

	## Till here

	fig=plt.figure(figsize=(8,8))  #showing color map using histogram plot
	fig.add_subplot(5,2,1)
	plt.imshow(Image.fromarray(img_barba_res.astype('uint8')),cmap='gray')
	plt.colorbar()
	fig.add_subplot(5,2,2)
	plt.imshow(img_barba,cmap='gray')
	plt.colorbar()
	fig.add_subplot(5,2,3)
	plt.imshow(Image.fromarray(img_canyon_res.astype('uint8')))
	plt.colorbar()
	fig.add_subplot(5,2,4)
	plt.imshow(img_canyon)
	plt.colorbar()
	fig.add_subplot(5,2,5)
	plt.imshow(Image.fromarray(img_tem_res.astype('uint8')),cmap='gray')
	plt.colorbar()
	fig.add_subplot(5,2,6)
	plt.imshow(img_tem,cmap='gray')
	plt.colorbar()
	fig.add_subplot(5,2,7)
	plt.imshow(Image.fromarray(img_church_res.astype('uint8')))
	plt.colorbar()
	fig.add_subplot(5,2,8)
	plt.imshow(img_church)
	plt.colorbar()
	fig.add_subplot(5,2,9)
	plt.imshow(Image.fromarray(img_retina_res.astype('uint8')))
	plt.colorbar()
	fig.add_subplot(5,2,10)
	plt.imshow(img_retina)
	plt.colorbar()
	plt.show()

	

	

	

	

	

