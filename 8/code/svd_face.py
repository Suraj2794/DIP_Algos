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
						test_labels.append(no)
				no+=1
			else:
				for f in range(0,len(fileName)):
					fullPath=dPath+'/'+fileName[f]
					img=Image.open(fullPath)
					img=np.array(img,dtype='float')
					img=img/np.amax(img)
					train_images.append(img)
					img=img.reshape(img.shape[0]*img.shape[1],)
					unseen_images.append(img)

	return train_images,test_images,column_images,column_test_images,train_labels,test_labels,unseen_images




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
	train_images,test_images,column_images,column_test_images,train_labels,test_labels,unseen_images=read_images('att_faces/')
	column_images=np.array(column_images,dtype='float')
	column_test_images=np.array(column_test_images,dtype='float')
	unseen_images=np.array(unseen_images,dtype='float')
	sum_column=sum(column_images[:len(column_images)])/len(column_images)
	sum_column_test=sum(column_test_images[:len(column_test_images)])/len(column_test_images)
	sum_unseen_images=sum(unseen_images[:len(unseen_images)])/len(unseen_images)
	diff_column=column_images-sum_column
	diff_column_test=column_test_images-sum_column_test
	diff_unseen_images=unseen_images-sum_unseen_images
	L=np.dot(diff_column,diff_column.T)
	eig_L=np.linalg.svd(L)[0]
	x=[]
	y=[]
	for k in (1,2,3,5,10,15,20,30,50,75,100,150,170):
		c_eig=np.dot(diff_column.T,eig_L.T[:k].T)
		norm_c_eig=np.divide(c_eig.T,np.linalg.norm(c_eig.T,2,1).reshape(1,k).T)
		train_eig_val=np.dot(diff_column,norm_c_eig.T)
		test_eig=np.dot(diff_column_test,norm_c_eig.T)
		unseen_eig=np.dot(diff_unseen_images,norm_c_eig.T)
		acc=accuracy(test_eig,train_eig_val,train_labels,test_labels)*100
		print(k,',',acc)
		x.append(k)
		y.append(acc)




	plt.plot(x,y)
	plt.xlabel('K')
	plt.ylabel('accuracy')
	plt.suptitle('k vs accuracy graph (svd att)')
	plt.savefig('k_acc_att_svd.png')
	plt.show()
	#plt.close()




