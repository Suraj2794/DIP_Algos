import numpy as np
from PIL import Image  #This library is used to show and load images (imp)
from matplotlib import pyplot as plt

def adaptive_hist(img_arr,window_size):  #function for calculating Contrast Limited Adaptive histogram equalization
	img_res=np.copy(img_arr)
	window=np.zeros(window_size) #creating window array of user specified size
	window_center_r=int(window_size[0]/2) #calculating center of the window
	window_center_c=int(window_size[1]/2)
	(x,y)=img_arr.shape
	for i in range(x):
		for j in range(y):
			window=np.zeros_like(window,dtype=int) #getting window of user specified size with image pixels in it and placing image pixel at center
			#print(i,j,window_center_r,window_center_r+min(window_size[0]-window_center_r,x-i),window_center_c,window_center_c+min(window_size[0]-window_center_c,y-j))
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_arr[i:(i+min(x-i,window_size[0]-window_center_r)),j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c-min(j,window_center_c):window_center_c+1]=img_arr[i:(i+min(x-i,window_size[0]-window_center_r)),(j-min(j,window_center_c)):j+1]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_arr[i-min(i,window_center_r):i+1,j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c-min(j,window_center_c):window_center_c+1]=img_arr[i-min(i,window_center_r):i+1,(j-min(j,window_center_c)):j+1]
			a=np.nonzero(window)  #getting the coordinates where non zero elements starts in window
			x_s=a[0][0]
			y_s=a[1][0]
			x_e=a[0][len(a[0])-1]
			y_e=a[1][len(a[1])-1]
			#print(window)
			cf=hist_eq_1(window[x_s:x_e+1,y_s:y_e+1],0.6) #calculating histogram equalization of window using clipping value of 0.6
			if cf != None:
				img_res[i,j]=cf[img_arr[i,j]]*256   #giving pixel values
		#print(i)
	return img_res



def adaptive_hist_col(img_arr,window_size): #function for calculating Adaptive histogram equalization for color images
	img_res=np.copy(img_arr)      #this is exactly same as above function except it works on all three channels
	window=np.zeros(window_size)
	window_center_r=int(window_size[0]/2)
	window_center_c=int(window_size[1]/2)
	(x,y,z)=img_arr.shape
	for k in range(z):
		img_arr1=img_arr[:,:,k]
		for i in range(x):
			for j in range(y):
				window=np.zeros_like(window,dtype=int)
			#print(i,j,window_center_r,window_center_r+min(window_size[0]-window_center_r,x-i),window_center_c,window_center_c+min(window_size[0]-window_center_c,y-j))
				window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_arr1[i:(i+min(x-i,window_size[0]-window_center_r)),j:(j+min(window_size[1]-window_center_c,y-j))]
				window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c-min(j,window_center_c):window_center_c+1]=img_arr1[i:(i+min(x-i,window_size[0]-window_center_r)),(j-min(j,window_center_c)):j+1]
				window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=img_arr1[i-min(i,window_center_r):i+1,j:(j+min(window_size[1]-window_center_c,y-j))]
				window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c-min(j,window_center_c):window_center_c+1]=img_arr1[i-min(i,window_center_r):i+1,(j-min(j,window_center_c)):j+1]
				a=np.nonzero(window)
				x_s=a[0][0]
				y_s=a[1][0]
				x_e=a[0][len(a[0])-1]
				y_e=a[1][len(a[1])-1]
				cf=hist_eq_1(window[x_s:x_e+1,y_s:y_e+1],0.6)
				if cf != None:
					img_res[i,j,k]=cf[img_arr[i,j,k]]*256
		#print(i)
	return img_res


def hist_eq_1(img_arr, clip): # this function used to calculate histogram equilization of given window
	img_res=np.copy(img_arr)
	hist_data=[0]*256
	#print(img_arr.shape) 
	for i in range(img_arr.shape[0]):  # calculating hist data for all intensity values of a given window
		for j in range(img_arr.shape[1]):
			hist_data[img_arr[i,j]]+=1

	(a,b)=img_arr.shape
	c=a*b
	if c!=0:
		hist_data=[x/float(c) for x in hist_data]
		hist_data = [clip if x > clip else x for x in hist_data] #calculating hist_data using clipping parameter
		delta = (1-sum(hist_data)) / c
		hist_data = [x + delta for x in hist_data]
		#print(sum(hist_data))
	else:
		return None
	cf_sum=[0]*256
	for i in range(len(hist_data)):
		cf_sum[i]=sum(hist_data[:i+1])
	#print(cf_sum)
	if img_arr.shape[0]==0 or img_arr.shape[1]==0:
		return None
	cf_p=[cf_sum[i] for i in range(len(cf_sum))] #calculating cdf 
	
	#for i in range(img_arr.shape[0]):
	 #   for j in range(img_arr.shape[1]):
	  #	  img_res[i,j]=256*cf_p[img_arr[i,j]]
	#print(img_res)
	return cf_p


if __name__ == '__main__': 
	img_barba=Image.open("../data/barbara.png")  #Opening all given images using PIL
	img_tem=Image.open("../data/TEM.png")
	img_canyon=Image.open("../data/canyon.png")
	img_church=Image.open("../data/church.png")
	img_retina=Image.open("../data/retina.png")
	img_tem_arr=np.array(img_tem)  #converting all PIL images into numpy nd array
	img_canyon_arr=np.array(img_canyon)
	img_barba_arr=np.array(img_barba)
	img_church_arr=np.array(img_church)
	img_retina_arr=np.array(img_retina)
	#img_barba_hist=adaptive_hist(img_barba_arr,(51,51))
	if(len(img_barba_arr.shape)) ==2: ## checking for colored images
		img_barba_res=adaptive_hist(img_barba_arr,(101,101)) #getting adaptive histogram equalized image with given window size passed as a parameter
	elif (len(img_barba_arr.shape)) == 3:
		img_barba_res=adaptive_hist_col(img_barba_arr,(51,51))
	
	img_barba.show() #showing original image
	Image.fromarray(img_barba_res.astype('uint8')).show() ##showing histogram equalized image

	## Same for all the below images

	if(len(img_tem_arr.shape))==2:
		img_tem_res=adaptive_hist(img_tem_arr,(101,101))
	elif (len(img_tem_arr.shape)) == 3:
		img_tem_res=adaptive_hist_col(img_tem_arr,(101,101))

	img_tem.show()
	Image.fromarray(img_tem_res.astype('uint8')).show()

	if(len(img_canyon_arr.shape)) ==2:
		img_canyon_res=adaptive_hist(img_canyon_arr,(51,51))
	elif (len(img_canyon_arr.shape)) == 3:
		img_canyon_res=adaptive_hist_col(img_canyon_arr,(51,51))

	img_canyon.show()
	Image.fromarray(img_canyon_res.astype('uint8')).show()

	if(len(img_church_arr.shape)) ==2:
		img_church_res=adaptive_hist(img_church_arr,(51,51))
	elif (len(img_church_arr.shape)) == 3:
		img_church_res=adaptive_hist_col(img_church_arr,(51,51))

	img_church.show()
	Image.fromarray(img_church_res.astype('uint8')).show()

	if(len(img_retina_arr.shape)) ==2:
		img_retina_res=adaptive_hist(img_retina_arr,(51,51))
	elif (len(img_retina_arr.shape)) == 3:
		img_retina_res=adaptive_hist_col(img_retina_arr,(51,51))

	img_retina.show()
	Image.fromarray(img_retina_res.astype('uint8')).show()


		#till here
	