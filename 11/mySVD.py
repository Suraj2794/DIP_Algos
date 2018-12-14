from os import listdir,walk
from os.path import isfile, join
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def create_SVD(inp_arr):
    U=np.dot(inp_arr,inp_arr.T)
    V=np.dot(inp_arr.T,inp_arr)
    U_eig=np.linalg.eig(U)
    V_eig=np.linalg.eig(V)
    tmp=np.dot(inp_arr,V_eig[1])
    s=np.dot(U_eig[1].T,tmp)
    return U_eig[1],s.real,V_eig[1].T


if __name__ == '__main__':
	imgPath='att_faces/s1/1.pgm'
	img=Image.open(imgPath)
	img=np.array(img,dtype='float')
	img=img/np.amax(img)

	u,s,v=create_SVD(img)

	# Cross checking

	temp=np.dot(s,v)
	recon_arr=np.dot(u,temp).real

	fig=plt.figure(figsize=(8,8))  #showing color map using histogram plot
	fig.add_subplot(1,2,1)
	plt.imshow(img,cmap='gray')
	plt.colorbar()
	fig.add_subplot(1,2,2)
	plt.imshow(recon_arr,cmap='gray')
	plt.colorbar()
	plt.suptitle('Left Side input Image array passed and right side image array Reconstructed from SVD')
	plt.show()


