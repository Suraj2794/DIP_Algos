from os import listdir,walk
from os.path import isfile, join
from PIL import Image
import numpy as np
import sys
from matplotlib import pyplot as plt

def read_images(basePath):

	train_images=[]
	test_images=[]
	column_test_images=[]
	column_images=[]
	train_labels=[]
	test_labels=[]
	no=0

def read_Yale(basePath):
	train_images=[]
	test_images=[]
	column_test_images=[]
	column_images=[]
	train_labels=[]
	test_labels=[]
	no=0
	for dPath,dName,fileName in walk('CroppedYale/'):
		if len(dName) == 0:
			for f in range(0,len(fileName)):
				if(f < 41):
					fullPath=dPath+'/'+fileName[f]
					img=Image.open(fullPath)
					img=np.array(img,dtype='float')
					img=img/np.amax(img)
					train_images.append(img)
					img=img.reshape(img.shape[0]*img.shape[1],)
					column_images.append(img)
					train_labels.append(no)
				else:
					fullPath=dPath+'/'+fileName[f]
					img=Image.open(fullPath)
					img=np.array(img,dtype='float')
					img=img/np.amax(img)
					test_images.append(img)
					img=img.reshape(img.shape[0]*img.shape[1],)
					column_test_images.append(img)
					test_labels.append(no)
			no+=1
	return train_images,test_images,column_images,column_test_images,train_labels,test_labels


def accuracy(test_eig,train_eig_val,train_labels,test_labels):
	no=0
	results=[]
	accuracy=0
	for t in test_eig:
		diff_tr_tt=train_eig_val-t
		a=np.linalg.norm(diff_tr_tt,2,1)
		if train_labels[np.argmin(a)] == test_labels[no]:
			accuracy+=1
		no+=1
	return accuracy/no

if __name__ == '__main__':
	train_images,test_images,column_images,column_test_images,train_labels,test_labels=read_Yale('CroppedYale/')
	column_images=np.array(column_images,dtype='float')
	column_test_images=np.array(column_test_images,dtype='float')
	sum_column=sum(column_images[:len(column_images)])/len(column_images)
	sum_column_test=sum(column_test_images[:len(column_test_images)])/len(column_test_images)
	diff_column=column_images-sum_column
	diff_column_test=column_test_images-sum_column_test
	L=np.dot(diff_column,diff_column.T)
	eig_L=np.linalg.svd(L)[0]
	fig=plt.figure(figsize=(30,30))
	sub_no=1
	for k in (2,10,20,50,75,100,125,150,175):
		c_eig=np.dot(diff_column.T,eig_L.T[:k].T)
		norm_c_eig=np.divide(c_eig.T,np.linalg.norm(c_eig.T,2,1).reshape(1,k).T)
		img_eig_val=np.dot(diff_column[0],norm_c_eig.T)
		recon_img=np.sum(np.multiply(norm_c_eig[:k],img_eig_val.reshape(k,1)),0).reshape(192,168)
		fig.add_subplot(3,3,sub_no)
		plt.imshow(recon_img,cmap='gray')
		plt.colorbar()
		sub_no+=1
	plt.show()
	fig.savefig('recon_yale.png')
	fig=plt.figure(figsize=(30,30))
	c_eig=np.dot(diff_column.T,eig_L.T[:25].T)
	norm_c_eig=np.divide(c_eig.T,np.linalg.norm(c_eig.T,2,1).reshape(1,25).T)
	for i in range(0,25):
		fig.add_subplot(5,5,i+1)
		plt.imshow(norm_c_eig[i].reshape(192,168),cmap='gray')
		plt.colorbar()
	plt.show()
	fig.savefig('25-eigenfaces.png')





