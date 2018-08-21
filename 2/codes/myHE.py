import numpy as np
from PIL import Image   ##This library is used to show and load images (imp)
from matplotlib import pyplot as plt

def hist_eq(img_arr): # histogram equilizer for GrayScale images
	img_res=np.copy(img_arr)   
	hist_data=[0]*256   # calculating hist data for all intensity values in a Gray scale image
	for i in range(img_arr.shape[0]):
		for j in range(img_arr.shape[1]):
				hist_data[img_arr[i,j]]+=1

 
	cf_sum=[0]*256   # calculating its CDF
	for i in range(len(hist_data)):
		 cf_sum[i]=sum(hist_data[:i+1])

	cf_p=[cf_sum[i]/(512*512) for i in range(len(cf_sum))]  # calculating probabilites
	for i in range(img_arr.shape[0]):
		 for j in range(img_arr.shape[1]):
				img_res[i,j]=256*cf_p[img_arr[i,j]]  #giving pixel value to  new_image based on cdf of given image as input
	return img_res


def hist_eq_color(img_arr): #histogram equalizer for colored images
	img_res=np.copy(img_arr) 
	hist_data=[[0]*256,[0]*256,[0]*256]  # calculating hist data for all intensity values for all the channels in a Colored Image
	for i in range(img_arr.shape[0]):
		 for j in range(img_arr.shape[1]):
				for k in range(img_arr.shape[2]):
					hist_data[k][img_arr[i,j,k]]+=1


	cf_sum=[[0]*256,[0]*256,[0]*256]  #calculating CDF for all the channels of an Image
	for i in range(len(cf_sum)):
		 for j in range(len(hist_data[i])):
				cf_sum[i][j]=sum(hist_data[i][:j+1])
	
	#print(cf_sum)
	cf_p=[[cf_sum[j][i]/(img_arr.shape[0]*img_arr.shape[1]) for i in range(len(cf_sum[j]))] for j in range(len(cf_sum))] #finding probabilities of CDF for all the channels
	#print(cf_p)
	for i in range(img_arr.shape[0]):
		 for j in range(img_arr.shape[1]):
				for k in range(img_arr.shape[2]):
					img_res[i,j,k]=256*cf_p[k][img_arr[i,j,k]]  #giving pixel value to  new_image based on cdf of given image as input
	return img_res

if __name__ == '__main__':
	img_barba=Image.open("../data/barbara.png")   #Opening all given images using PIL
	img_tem=Image.open("../data/TEM.png")
	img_canyon=Image.open("../data/canyon.png")
	img_church=Image.open("../data/church.png")
	img_retina=Image.open("../data/retina.png")
	img_tem_arr=np.array(img_tem)    #converting all PIL images into numpy nd array
	img_canyon_arr=np.array(img_canyon)
	img_barba_arr=np.array(img_barba)
	img_church_arr=np.array(img_church)
	img_retina_arr=np.array(img_retina)
	if(len(img_barba_arr.shape)) ==2:    # checking for colored images
		img_barba_res=hist_eq(img_barba_arr)  #getting histogram equalized image
	elif (len(img_barba_arr.shape)) == 3:
		img_barba_res=hist_eq_color(img_barba_arr)
	
	img_barba.show()  #showing original image
	Image.fromarray(img_barba_res.astype('uint8')).show() #showing histogram equalized image

	## Same for all the below images

	if(len(img_tem_arr.shape)) ==2:
		img_tem_res=hist_eq(img_tem_arr)
	elif (len(img_tem_arr.shape)) == 3:
		img_tem_res=hist_eq_color(img_tem_arr)

	img_tem.show()
	Image.fromarray(img_tem_res.astype('uint8')).show()

	if(len(img_canyon_arr.shape)) ==2:
		img_canyon_res=hist_eq(img_canyon_arr)
	elif (len(img_canyon_arr.shape)) == 3:
		img_canyon_res=hist_eq_color(img_canyon_arr)

	img_canyon.show()
	Image.fromarray(img_canyon_res.astype('uint8')).show()

	if(len(img_church_arr.shape)) ==2:
		img_church_res=hist_eq(img_church_arr)
	elif (len(img_church_arr.shape)) == 3:
		img_church_res=hist_eq_color(img_church_arr)

	img_church.show()
	Image.fromarray(img_church_res.astype('uint8')).show()

	if(len(img_retina_arr.shape)) ==2:
		img_retina_res=hist_eq(img_retina_arr)
	elif (len(img_retina_arr.shape)) == 3:
		img_retina_res=hist_eq_color(img_retina_arr)

	img_retina.show()
	Image.fromarray(img_retina_res.astype('uint8')).show()

	#Till here

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

	

	

	

	

	

