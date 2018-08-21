import numpy as np
from PIL import Image ##This library is used to show and load images (imp)
from matplotlib import pyplot as plt

def hist_eq_color(img_arr):  #histogram equalizer for colored images
	img_res=np.copy(img_arr)
	hist_data=[[0]*256,[0]*256,[0]*256] # calculating hist data for all intensity values for all the channels in a Colored Image
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
					img_res[i,j,k]=256*cf_p[k][img_arr[i,j,k]] #giving pixel value to  new_image based on cdf of given image as input
	return img_res

def hist_match_color(img_ref_arr,img_inp_arr):  # function for calculating histogram matched image with refrence image
	img_res=np.copy(img_inp_arr)   
	hist_ref_data=[[0]*256,[0]*256,[0]*256]  # calculate histogram equalized image for input and refrenced images
	hist_inp_data=[[0]*256,[0]*256,[0]*256]
	for i in range(img_ref_arr.shape[0]):
		for j in range(img_ref_arr.shape[1]):
			for k in range(img_ref_arr.shape[2]):
				hist_ref_data[k][img_ref_arr[i,j,k]]+=1
				
	for i in range(img_inp_arr.shape[0]):
		for j in range(img_inp_arr.shape[1]):
			for k in range(img_inp_arr.shape[2]):
				hist_inp_data[k][img_inp_arr[i,j,k]]+=1
				
				
	cf_ref_sum=[[0]*256,[0]*256,[0]*256]
	for i in range(len(cf_ref_sum)):
		for j in range(len(hist_ref_data[i])):
			cf_ref_sum[i][j]=sum(hist_ref_data[i][:j+1])

	cf_inp_sum=[[0]*256,[0]*256,[0]*256]
	for i in range(len(cf_inp_sum)):
		for j in range(len(hist_inp_data[i])):
			cf_inp_sum[i][j]=sum(hist_inp_data[i][:j+1])
	
	inp_pixe=[[0]*256,[0]*256,[0]*256]  # here histogram matching is performed
	for i in range(len(cf_ref_sum)):
		for j in range(len(cf_inp_sum[i])):
			for k in range(j,len(cf_ref_sum[i])):
				#print(i,j,k,cf_inp_sum[i][j],cf_ref_sum[i][k])
				if (cf_inp_sum[i][j]/cf_ref_sum[i][k] > 0.85  and cf_inp_sum[i][j]/cf_ref_sum[i][k] <= 1):# or (cf_ref_sum[i][j]/cf_inp_sum[i][k] > 0.85 and cf_ref_sum[i][j]/cf_inp_sum[i][k] <= 1):
					#print(i,j,k,cf_inp_sum[i][j],cf_ref_sum[i][k])
					inp_pixe[i][j]=k
					break;
	#print(cf_inp_sum,'----------',cf_ref_sum)
	for i in range(img_inp_arr.shape[0]):
		for j in range(img_inp_arr.shape[1]):
			for k in range(img_inp_arr.shape[2]):
				#print(i,j,k,img_inp_arr[i,j,k])
				img_res[i,j,k]=inp_pixe[k][img_inp_arr[i,j,k]] # after performing histogram matching new pixel values allocated to the output image
	return img_res



if __name__ == '__main__':
	img_eye_inp=Image.open("../data/retina.png") #Opening all given images using PIL 
	img_eye_ref=Image.open("../data/retinaRef.png")
	img_eye_inp_arr=np.array(img_eye_inp) #converting all PIL images into numpy nd array
	img_eye_ref_arr=np.array(img_eye_ref)
	img_eye_res=hist_match_color(img_eye_ref_arr,img_eye_inp_arr) # histogram matched image
	img_eye_heq=hist_eq_color(img_eye_inp_arr) #histogram equalized image
	img_eye_inp.show("Original Image") #show original image
	Image.fromarray(img_eye_res.astype('uint8')).show()  # show histogram matched image
	#Image.fromarray(img_eye_heq.astype('uint8')).show()
	fig=plt.figure(figsize=(8,8))  #showing colored map for ouput,input,refrenced images
	fig.add_subplot(3,1,1)
	plt.imshow(img_eye_inp)
	plt.colorbar()
	fig.add_subplot(3,1,2)
	plt.imshow(Image.fromarray(img_eye_res.astype('uint8')))
	plt.colorbar()
	fig.add_subplot(3,1,3)
	plt.imshow(Image.fromarray(img_eye_heq.astype('uint8')))
	plt.colorbar()
	plt.show()
