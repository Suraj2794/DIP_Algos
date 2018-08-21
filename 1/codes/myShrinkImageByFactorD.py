import numpy as np
from PIL import Image   ##This library is used to show images (imp)
from matplotlib import pyplot as plt

def shrink_image_d(img_arr,d):
	shrink_img=Image.fromarray(img_arr[::d][:,::d].astype('uint8'))  #select alternate d row and column of an image
	return shrink_img


if __name__ == '__main__':
	img_circle=Image.open("../data/circles_concentric.png")  # Open image using PIL library
	img_circle.show()
	shrink_image1=shrink_image_d(np.array(img_circle),2) # compressing image by factor 2
	shrink_image2=shrink_image_d(np.array(img_circle),3) # compressing image by factor 3
	shrink_image1.show()
	shrink_image2.show()
	fig=plt.figure(figsize=(8,8))
	fig.add_subplot(2,1,1)
	plt.imshow(shrink_image1,cmap='gray')
	plt.colorbar()
	fig.add_subplot(2,1,2)
	plt.imshow(shrink_image2,cmap='gray')
	plt.colorbar()
	plt.show()