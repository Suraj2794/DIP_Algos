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
	x=[]
	y=[]
	sub_no=0
	cl=["C0","C1"]
	#ig=plt.figure(figsize=(8,8))
	for z in (3,0):
		sub_no+=1
		for k in (1,2,3,5,10,15,20,30,50,60,65,75,100,200,300,500,1000):
			if(z < k):
				c_eig=np.dot(diff_column.T,eig_L.T[z:k].T)
				norm_c_eig=np.divide(c_eig.T,np.linalg.norm(c_eig.T,2,1).reshape(1,k-z).T)
				train_eig_val=np.dot(diff_column,norm_c_eig.T)
				test_eig=np.dot(diff_column_test,norm_c_eig.T)
				#unseen_eig=np.dot(diff_unseen_images,norm_c_eig.T)
				acc=accuracy(test_eig,train_eig_val,train_labels,test_labels)*100
				print(k,',',acc,'leaving ',z,' values')
				x.append(k)
				y.append(acc)
		plt.plot(x,y)
		plt.xlabel('K')
		plt.ylabel('accuracy')
		plt.suptitle('k vs accuracy graph (SVD Yal)~ leaving '+str(z)+' values')
		plt.savefig('k_acc_yal_svd_'+str(z)+'.png')
		plt.close()





