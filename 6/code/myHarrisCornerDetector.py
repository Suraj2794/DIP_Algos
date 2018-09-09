import numpy as np					  
from matplotlib import pyplot as plt
import hdf5storage
import math
import cv2 as cv

def finding_edge(img_array,operator):  #function to find edges basic operation used is convolution with given operator. Operator can be sobel or Perwitt operators
	window_size=operator.shape
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
			filtered_image[i,j]=find_window_edge(window,operator)
	return filtered_image 



def find_window_edge(window,operator):   # this is a function used by above function to calculate derivative in a given window	 
	edge_sum=0
	for i in range(window.shape[0]):
		for j in range(window.shape[1]):
			edge_sum+=window[i,j]*operator[i,j]
	return edge_sum

def get_gauss_kernel(size=3,sigma=1):  # returns gaussian kernel
	center=(int)(size/2)
	kernel=np.zeros((size,size))
	for i in range(size):
		for j in range(size):
			diff=np.sqrt((i-center)**2+(j-center)**2)
			kernel[i,j]=np.exp(-(diff**2)/2*sigma**2)
	return kernel/np.sum(kernel)

def apply_blur(img_array,kernel): 
	window_size=kernel.shape
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
			filtered_image[i,j]=perform_conv(window,kernel)
	return filtered_image 

def perform_conv(window,kernel):
	blur=0
	for i in range(window.shape[0]):
		for j in range(window.shape[1]):
			blur+=window[i,j]*kernel[i,j]
	return blur


def find_corner(img_array,grad_img_x,grad_img_y,kernel_size,k,sigma,threshold):  #function to find corners in a given image
	corner_list=[]
	grad_x_sq=grad_img_x**2
	grad_y_sq=grad_img_y**2
	grad_xy=grad_img_x*grad_img_y
	kernel=get_gauss_kernel(kernel_size,sigma)
	window_size=kernel.shape
	window=np.zeros(window_size,dtype=float)	#creating window array of user specified size
	window_center_r=int(window_size[0]/2) #calculating center of the window
	window_center_c=int(window_size[1]/2)
	#corn_img=np.zeros_like(img_array)
	corn_img=np.dstack([img_array,img_array,img_array])
	img_eig_min=np.ones_like(img_array)
	img_eig_max=np.ones_like(img_array)
	corn_score_val=np.ones_like(img_array)
	(x,y)=img_array.shape
	for i in range(x):
		for j in range(y):


			## getting sum for window grad_x
			window=np.zeros(window_size,dtype=float)  #getting window of user specified size with image pixels in it and placing image pixel at center
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=grad_x_sq[i:(i+min(x-i,window_size[0]-window_center_r)),j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c-min(j,window_center_c):window_center_c+1]=grad_x_sq[i:(i+min(x-i,window_size[0]-window_center_r)),(j-min(j,window_center_c)):j+1]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=grad_x_sq[i-min(i,window_center_r):i+1,j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c-min(j,window_center_c):window_center_c+1]=grad_x_sq[i-min(i,window_center_r):i+1,(j-min(j,window_center_c)):j+1]
			sum_x=perform_conv(window,kernel)
			
			
			##getting sum for windows grad_y
			window=np.zeros(window_size,dtype=float)  #getting window of user specified size with image pixels in it and placing image pixel at center
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=grad_y_sq[i:(i+min(x-i,window_size[0]-window_center_r)),j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c-min(j,window_center_c):window_center_c+1]=grad_y_sq[i:(i+min(x-i,window_size[0]-window_center_r)),(j-min(j,window_center_c)):j+1]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=grad_y_sq[i-min(i,window_center_r):i+1,j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c-min(j,window_center_c):window_center_c+1]=grad_y_sq[i-min(i,window_center_r):i+1,(j-min(j,window_center_c)):j+1]
			sum_y=perform_conv(window,kernel)
			
			## getting sum for window grad_xy
			window=np.zeros(window_size,dtype=float)  #getting window of user specified size with image pixels in it and placing image pixel at center
			#print(i,j,window_center_r,window_center_r+min(window_size[0]-window_center_r,x-i),window_center_c,window_center_c+min(window_size[0]-window_center_c,y-j))
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=grad_xy[i:(i+min(x-i,window_size[0]-window_center_r)),j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r:(window_center_r+min(window_size[0]-window_center_r,x-i)),window_center_c-min(j,window_center_c):window_center_c+1]=grad_xy[i:(i+min(x-i,window_size[0]-window_center_r)),(j-min(j,window_center_c)):j+1]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c:(window_center_c+min(window_size[0]-window_center_c,y-j))]=grad_xy[i-min(i,window_center_r):i+1,j:(j+min(window_size[1]-window_center_c,y-j))]
			window[window_center_r-min(i,window_center_r):window_center_r+1,window_center_c-min(j,window_center_c):window_center_c+1]=grad_xy[i-min(i,window_center_r):i+1,(j-min(j,window_center_c)):j+1]
			sum_xy=perform_conv(window,kernel)

			#creating array 
			eig_arr=np.array([[sum_x,sum_xy],[sum_xy,sum_y]])
			eig_vals=np.linalg.eigvals(eig_arr)
			img_eig_min[i,j]=np.amin(eig_vals)
			img_eig_max[i,j]=np.amax(eig_vals)
			det=sum_x*sum_y-sum_xy**2
			trace=sum_x+sum_y
			corner=det-k*(trace**2)
			corn_score_val[i,j]=corner
			if corner > threshold:
				corner_list.append([i,j,corner])
				corn_img[i,j,:]=[1,0,0]
				#print(corner,i,j,trace,det)
	return corner_list,corn_img,img_eig_max,img_eig_min,corn_score_val


def non_max_supress(img_array,kernel):
	window_size=kernel
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
			filtered_image[i,j]=np.amax(window)
	return filtered_image 



if __name__ == '__main__':
	boat_mat = hdf5storage.loadmat('../data/boat.mat') # reading mat image
	boat_array=np.array(boat_mat['imageOrig']) #converting it to numpy array
	boat_array=boat_array/np.amax(boat_array) # normalizing image with in range [0,1]
	edge_boat_vr=finding_edge(boat_array,np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))  #finding vertical edge
	edge_boat_hr=finding_edge(boat_array,np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))  #finding horizontal edge

	corners,corn_img,eig_max,eig_min,corn_score=find_corner(boat_array,edge_boat_hr,edge_boat_vr,3,0.01,0.3,0.3)  # calculating corners in image
	corn_non=non_max_supress(corn_score,(3,3)) 
	#saving all images in image folder

	plt.imsave('../images/boat_corner_img.png',corn_img/np.amax(corn_img),cmap='gray')
	plt.imsave('../images/real.png',boat_array,cmap='gray')
	plt.imsave('../images/eigen_max.png',eig_max/np.amax(eig_max),cmap='gray')
	plt.imsave('../images/eigen_min.png',eig_min/np.amax(eig_min),cmap='gray')
	plt.imsave('../images/boat_hr.png',edge_boat_hr/np.amax(edge_boat_hr),cmap='gray')
	plt.imsave('../images/boat_vr.png',edge_boat_vr/np.amax(edge_boat_vr),cmap='gray')
	plt.imsave('../images/corn_non.png',corn_non/np.amax(corn_non),cmap='gray')
	plt.imsave('../images/corn_score.png',corn_score/np.amax(corn_score),cmap='gray')
	
	print("Number of corners found:--",len(corners))
	print("Corner list \n",corners)

	fig=plt.figure(figsize=(30,30))
	fig.add_subplot(1,2,1)
	plt.imshow(edge_boat_vr/np.amax(edge_boat_vr),cmap='gray')
	plt.title("Vertical edges")
	plt.colorbar()
	fig.add_subplot(1,2,2)
	plt.imshow(edge_boat_hr/np.amax(edge_boat_hr),cmap='gray')
	plt.title("Horizontal edges")
	plt.colorbar()

	fig1=plt.figure(figsize=(30,30))
	fig1.add_subplot(1,2,1)
	plt.imshow(eig_max/np.amax(eig_max),cmap='gray')
	plt.title("Eigen Max")
	plt.colorbar()
	fig1.add_subplot(1,2,2)
	plt.imshow(eig_min/np.amax(eig_min),cmap='gray')
	plt.title("Eigen Min")
	plt.colorbar()

	fig3=plt.figure(figsize=(30,30))
	fig3.add_subplot(1,2,1)
	plt.imshow(corn_score/np.amax(corn_score),cmap='gray')
	plt.title("Corner Score Image")
	plt.colorbar()
	fig3.add_subplot(1,2,2)
	plt.imshow(corn_non/np.amax(corn_non),cmap='gray')
	plt.title("Corner Non-Max supress Image")
	plt.colorbar()

	fig2=plt.figure(figsize=(30,30))
	fig2.add_subplot(1,2,1)
	plt.imshow(boat_array,cmap='gray')
	plt.title("Real Image")
	plt.colorbar()
	fig2.add_subplot(1,2,2)
	plt.imshow(corn_img/np.amax(corn_img),cmap='gray')
	plt.title("Corner Image")
	plt.colorbar()


	

	plt.show()