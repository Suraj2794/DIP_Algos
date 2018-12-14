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
	unseen_images=[]
	no=0

	for dPath,dName,fileName in walk('att_faces/'):
		if len(dName) == 0:
			if(int(dPath.split('/')[1].split('s')[1]) < 33):
				for f in range(0,len(fileName)):
					if(int(fileName[f].split('.')[0]) < 7):
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
						test_labels.append(1)
				no+=1
			else:
				for f in range(0,len(fileName)):
					fullPath=dPath+'/'+fileName[f]
					img=Image.open(fullPath)
					img=np.array(img,dtype='float')
					img=img/np.amax(img)
					train_images.append(img)
					img=img.reshape(img.shape[0]*img.shape[1],)
					column_test_images.append(img)
					test_labels.append(0)

	return train_images,test_images,column_images,column_test_images,train_labels,test_labels  #unseen_images



def accuracy(test_eig,train_eig_val,test_labels):
	no=0
	results=[]
	accuracy=0
	fp=0
	fn=0
	for t in test_eig:
		diff_tr_tt=train_eig_val-t
		a=np.linalg.norm(diff_tr_tt,2,1)
		if np.amin(a) > th:
			if test_labels[no]==1:
				fn+=1
		else:
			if test_labels[no] == 0:
				fp+=1
		no+=1
	return fn,fp,no

if __name__ == '__main__':
	k=50
	th=10
	train_images,test_images,column_images,column_test_images,train_labels,test_labels=read_images('att_faces/')
	column_images=np.array(column_images,dtype='float')
	column_test_images=np.array(column_test_images,dtype='float')
	#unseen_images=np.array(unseen_images,dtype='float')
	sum_column=sum(column_images[:len(column_images)])/len(column_images)
	sum_column_test=sum(column_test_images[:len(column_test_images)])/len(column_test_images)
	#sum_unseen_images=sum(unseen_images[:len(unseen_images)])/len(unseen_images)
	diff_column=column_images-sum_column
	diff_column_test=column_test_images-sum_column_test
	#diff_unseen_images=unseen_images-sum_unseen_images
	L=np.dot(diff_column,diff_column.T)
	eig_L=np.linalg.eig(L)
	c_eig=np.dot(diff_column.T,eig_L[1].T[:k].T)
	norm_c_eig=np.divide(c_eig.T,np.linalg.norm(c_eig.T,2,1).reshape(1,k).T)
	train_eig_val=np.dot(diff_column,norm_c_eig.T)
	test_eig=np.dot(diff_column_test,norm_c_eig.T)
	fn,fp,no=accuracy(test_eig,train_eig_val,test_labels)
	print('FalseNegative:-- ',fn)
	print('FalsePositive:--',fp)





