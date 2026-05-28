import pdb
import h5py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def createOutputAsCentroids(data,label):
	#pdb.set_trace()
	centroidLabels=np.unique(label)
	outputData=np.zeros([np.shape(data)[0],np.shape(data)[1]])
	for i in range(len(centroidLabels)):
		indices=np.where(label == centroidLabels[i])[0]
		tmpData=data[indices,:]
		centroid=np.mean(tmpData,axis=0)
		outputData[indices,]=centroid
	#pdb.set_trace()
	return outputData

def LCE(X,labels):
	#This is the implementation of PrincipalCentroidComponentAnalysis
	#Eigen decomposition is used to solve the optimization problem
	#data: [n x d] array where n: no of samples and d: no of features.
	#label: [n x 1] array or a list with m elements where m is no. of samples
	#pdb.set_trace()
	
	
	#mean centered the data
	X = X - np.mean(X,axis=0)

	#calculate the C matrix
	C = createOutputAsCentroids(X,labels)

	#transpose the matrices X, C to make each column a data point
	X,C = X.T,C.T

	#build the Q matrix using formulation 2
	#Q = np.dot(X,C.T) + np.dot(C,X.T) - np.dot(X,X.T)
	Q = 2*np.dot(C,X.T) - np.dot(X,X.T)
	
	eVals,eVecs = np.linalg.eigh(Q)
	#pdb.set_trace()

	#the eigen values are returned in ascending order.
	#Because we are maximizing the cost the eigen vector related to the largest eigenvalue needs to be picked.
	eVals,eVecs = np.flip(eVals),np.flip(eVecs,axis=1)
	return eVals,eVecs
	
def kNNAccuracy(D_trn,L_trn,D_tst,L_tst,N_NEIGHBORS=5,verbose=False):
	
	knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
	knn.fit(D_trn, L_trn.flatten())
	tstLabelsPred = knn.predict(D_tst)
	accuracy = 100 * accuracy_score(L_tst.flatten(), tstLabelsPred)
	if verbose:
		print('Accuracy on test data:',np.round(accuracy,2))
	return accuracy

	
def classifyWithSLCE(D,L,k,dataSetName,splitRatio,nItr):
	
	allACC_LCE = np.zeros(nItr)
	labelMap = {}
	#pdb.set_trace()
	for itr in range(nItr):
		#split data into train and test by 50:50 ratio	
		D_tr,D_tst,L_tr,L_tst = train_test_split(D, L, test_size=splitRatio, shuffle=True)
		#pdb.set_trace()
		#substract mean of training data from test
		trnMu = np.mean(D_tr,axis=0)
		D_tr = D_tr - trnMu
		D_tst = D_tst - trnMu
		
		# set the optimal dimension for SLCE
		p = len(np.unique(L_tr)) - 1
		
		
		#call LCE and calculate k-NN accuracy
		Ev,U = LCE(D_tr,L_tr)
		
		pDataTr = np.dot(U[:,:p].T,D_tr.T).T
		pDataTst = np.dot(U[:,:p].T,D_tst.T).T
		
		allACC_LCE[itr] = kNNAccuracy(pDataTr,L_tr,pDataTst,L_tst,k)
		print('Iteration',itr+1,str(k)+'-NN accuracy using SLCE:{:.2f}%'.format(allACC_LCE[itr]))
		
	print(str(k)+'-NN accuracy on dataset',dataSetName,'on',p,'-dimensional space')
	print('\t SLCE:{:.2f}%'.format(np.mean(allACC_LCE)),'+/- {:.2f}%'.format(np.std(allACC_LCE)))
	
def loadMouse_Bladder_CellData():
	f1 = h5py.File('./mouse_bladder_cell.h5','r+')
	D,L = np.array(f1['X']),np.array(f1['Y'])
	L = L.reshape(-1,1)
	return D,L


if __name__ == "__main__":
	
	dataSetName = 'Mouse_Bladder_Cell'
	
	#experimental specific set up
	splitRatio = 0.5
	k = 1
	nItr = 20
	D,L = loadMouse_Bladder_CellData()
	#pdb.set_trace()
	
	if dataSetName in ['10X_PBMC','Mouse_ES_Cell','Mouse_Bladder_Cell','Worm_Neuron_Cell']:
		#Run SVD() to reduce the dimension of data
		U,S,V = np.linalg.svd(D.T,full_matrices=False)#calling thin SVD
		D = np.dot(U.T,D.T).T
	
	classifyWithSLCE(D,L,k,dataSetName,splitRatio,nItr)
	
	

