## Created By: Prashant Shekhar

## Importing the packages

import numpy as np
from numpy import linalg as LA
from scipy import random, linalg
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import scipy as sp
import scipy.stats
from scipy.spatial import distance
import pickle


### Defining the functions

def Penalty_p(q,c):
    ## Objective: To generate the difference based penalty operator
    ## Input:
    ## 1. q: It is the order of difference which is being considered
    ## 2. c: This is the number of basis vectors under consideration
    ## Output:
    ## 1: P: The penalty matrix D^T.D
    ## This function allows to test the penalty till 4th order difference. In the paper we just use q = 1 or q = 2
    
    if q == 1:
        D = np.zeros([c-1,c])
        for i in range(c-1):
            D[i,i] = -1
            D[i,i+1] = 1
    if q == 2:
        D = np.zeros([c-2,c])
        for i in range(c-2):
            D[i,i]= 1
            D[i,i+1] = -2
            D[i,i+2] = 1
            
    if q == 3:
        D = np.zeros([c-3,c])
        for i in range(c-3):
            D[i,i] = -1
            D[i,i+1] = 3
            D[i,i+2] = -3
            D[i,i+3] = 1
    
    if q == 4:
        D = np.zeros([c-4,c])
        for i in range(c-4):
            D[i,i] = 1
            D[i,i+1] =-4
            D[i,i+2] = 4
            D[i,i+3] = -4
            D[i,i+3] = 1
    
    P = D.T.dot(D)
    return P




def Cost_func_1D(lam,Data,B,Pe):
    ## Objective: To compute the GCV fitting cost for univariate fitting
    ## Input:
    ## 1. lam: The smoothing hyperparameter
    ## 2. Data: The dataset to be modeled
    ## 3. B: The Basis matrix
    ## 4. Pe: The penalty matrix computed in function Penalty_p(q,c)
    ## Output:
    ## 1: GCV: The GCV metric value
    
    ### COmputing the Hat Matrix
    n = Data.shape[0]
    H = B.dot(np.linalg.inv(B.T.dot(B) + n*lam*Pe)).dot(B.T)
    ### GCV    
    r = (np.eye(H.shape[0]) - H).dot(Data[:,3].reshape(-1,1))
    numer = (1/n)*(np.linalg.norm(r))**2
    denom = ((1/n)*np.trace(np.eye(H.shape[0]) - H))**2
    obj = numer/denom
    return obj


def Cost_func_2D(lam,Data,B,Pe_x,Pe_y):
    ## Objective: To compute the GCV fitting cost for bivariate fitting
    ## Input:
    ## 1. lam: The smoothing hyperparameter [lam_x,lam_y]
    ## 2. Data: The dataset to be modeled
    ## 3. B: The Basis matrix
    ## 4. Pe_x: The penalty matrix in x direction
    ## 5. Pe_y: The penalty matrix in y direction
    ## Output:
    ## 1: GCV: The GCV metric value
    
    lam_x = lam[0]
    lam_y = lam[1]
    n = Data.shape[0]
    H = B.dot(np.linalg.inv(B.T.dot(B) + n*lam_x*Pe_x + n*lam_y*Pe_y)).dot(B.T)
    ### GCV    
    r = (np.eye(H.shape[0]) - H).dot(Data[:,3].reshape(-1,1))
    numer = (1/n)*(np.linalg.norm(r))**2
    denom = ((1/n)*np.trace(np.eye(H.shape[0]) - H))**2
    obj = numer/denom
    return obj





def Inference_1D(Data_temp,Data_pred,confidence = 0.95):
    ## Objective: To fit the hierarchical approach on univariate functions
    ## Input:
    ## 1. Data_temp: The dataset to be modeled
    ## 2. Data_pred: Prediction locations
    ## 3. confidence: % confidence level for the model
    ## Output:
    ## 1. results: The inference results for the prediction
    ## 2. Sparse: The sparse representation at each scale
    ## 3. qopt: Optimal penalty at each considered scale (either 1 or 2)
    ## 4. fun_vec: The cost of fitting at each scale
    
    ## Formatting the Data matrix
    if Data_temp.shape[1] == 3:
        Data = zeros([Data_temp.shape[0],4])
        Data[:,0] = Data_temp[:,0]
        Data[:,2] = Data_temp[:,1]
        Data[:,3] = Data_temp[:,2]
    
    if Data_temp.shape[1] == 4:
        Data = Data_temp
        

    ## Initializations
    points = Data.shape[0]
    Gaussian = np.empty([points,points])    
    ## Computing T
    dist = squareform(pdist(Data[:,0:2], 'euclidean'))
    max_d = np.amax(dist)
    k1 = 2*(max_d/2)**2;
    T = k1;
    l_s = 0
    s = 0
    n = Data.shape[0]
    tol = 1.0e9
    fun_vec = []
        
    results = {}
    qopt = []
    Sp = {}
    while l_s <= n:
        epsilon = T/(2**s)
        Gaussian = np.exp(-((dist**2)/epsilon))

        # Finding the rank of the Gaussian Kernel
        l_s = LA.matrix_rank(Gaussian)
        k = l_s+8

        # Getting Permutation ordering in terms of importance
        points = Data.shape[0]
        A = random.randn(k,points)
        W = A.dot(Gaussian)
        W = np.matrix(W) 
        Q,R,P = linalg.qr(W, mode='full', pivoting = True)   
                                                                                                              
        
        # Getting Perm: the permutation matrix
        Perm = np.zeros([len(P),len(P)],dtype=int)
        for w in range(0,len(P)):
            Perm[P[w],w] = 1     
                                
        
        # Getting sparse representation ordered by importance (Most important point at index 0)
        sparse = np.zeros([l_s,3])
        for i in range(l_s):
            sparse[i,0] = Data[P[i],0]
            sparse[i,1] = Data[P[i],1]
            sparse[i,2] = Data[P[i],3]
        
        
        # Selecting Relevant columns of Kernel into B Matrix
        Mat = Gaussian.dot(Perm)
        B = Mat[:,0:l_s];
        
        
        # Getting the Permutation operator for the coordinate vector which 
        # changes ordering of theta from decreasing order of importance to 
        # increasing sequentially (Relation <=)
        tt = sparse[:,0]
        tt_sort = np.argsort(tt)
        Permm = np.zeros([len(tt_sort),len(tt_sort)],dtype=int)
        for w in range(0,len(tt_sort)):
            Permm[w,tt_sort[w]] = 1

        
        
        # Computing the Penalty and Fitting the Regularization network
        q1 = 1
        Pe_temp = Penalty_p(q1,l_s)
        Pe = Permm.T.dot(Pe_temp).dot(Permm)
        args = (Data,B,Pe)
        bnds = [(1.0e-12, None)]
        par = [0.01]
        l = minimize(Cost_func_1D,par,args,bounds=bnds,method='SLSQP')
        lam1 = l.x[0]
        fun1 = l.fun

        
        q2 = 2
        Pe_temp = Penalty_p(q2,l_s)
        Pe = Permm.T.dot(Pe_temp).dot(Permm)
        args = (Data,B,Pe)
        bnds = [(1.0e-12, None)]
        par = [0.01]
        l = minimize(Cost_func_1D,par,args,bounds=bnds,method='SLSQP')
        lam2 = l.x[0]
        fun2 = l.fun
        
        if fun1<=fun2:
            fun_vec.append(fun1)
            lam = lam1
            optq = q1
        else:
            fun_vec.append(fun2)
            lam = lam2
            optq = q2
            
        #8. Returning the mean Prediction
        Bpred = np.zeros([Data_pred.shape[0],l_s])
        for i in range(Data_pred.shape[0]):
            for j in range(l_s):
                temp = distance.euclidean(Data_pred[i,:], np.array([sparse[j,0],sparse[j,1]]))
                Bpred[i,j]  = np.exp(-((temp**2)/epsilon))
        
        Pe_temp = Penalty_p(optq,l_s)
        Pe = Permm.T.dot(Pe_temp).dot(Permm)
        inver = np.linalg.inv(B.T.dot(B) + n*lam*Pe)
        theta = inver.dot(B.T.dot(Data[:,3].reshape(-1,1)))
        pred = Bpred.dot(theta)
        
        #9. Error bounds
        nr = (Data[:,3].reshape(-1,1) - B.dot(theta)).reshape(-1,1)
        term = B.dot(inver.dot(B.T))
        df_res = n - 2*np.trace(term) + np.trace(term.dot(term.T))
        sigmasq = (nr.T.dot(nr))/(df_res)
        sigmasq = sigmasq[0][0]
        std = np.sqrt(np.diag(sigmasq*Bpred.dot(inver).dot(Bpred.T)))
        stdev_t = sp.stats.t._ppf((1+confidence)/2.,df_res)*std
        results[s] = [pred,stdev_t]
        qopt.append(optq)
        Sp[s] = sparse
        print(s)
        if l_s == n:
            break
        s = s+1
    return [results,Sp,qopt,fun_vec]







def Inference_2D(Data,Data_pred,confidence = 0.95):
    ## Objective: To fit the hierarchical approach on univariate functions
    ## Input:
    ## 1. Data: The dataset to be modeled
    ## 2. Data_pred: Prediction locations
    ## 3. confidence: % confidence level for the model
    ## Output:
    ## 1. results: The inference results for the prediction
    ## 2. Sp: The sparse representation at each scale
    ## 3. qoptx: Optimal penalty at each considered scale (either 1 or 2) in x direction
    ## 4. qopty: Optimal penalty at each considered scale (either 1 or 2) in y direction
    ## 5. fun_vec: The cost of fitting at each scale

    
    ## Initializations
    points = Data.shape[0]
    Gaussian = np.empty([points,points])    
    ## Computing T
    dist = squareform(pdist(Data[:,0:2], 'euclidean'))
    max_d = np.amax(dist)
    k1 = 2*(max_d/2)**2;
    T = k1;
    l_s = 0
    s = 0
    n = Data.shape[0]
    tol = 1.0e9
    fun_vec = []
    results = {}
    qoptx = []
    qopty = []
    Sp = {}
    while l_s <= n:
        epsilon = T/(2**s)
        Gaussian = np.exp(-((dist**2)/epsilon))

        # Finding the rank of the Gaussian Kernel
        l_s = LA.matrix_rank(Gaussian)
        k = l_s+8

        #print('epsilon,T,s for this iteration is '+str(epsilon)+' '+str(T)+' '+str(s))

        # Calculating the W matrix
        points = Data.shape[0]
        A = random.randn(k,points)
        W = A.dot(Gaussian)
        W = np.matrix(W) 

        # Applying Pivoted QR on W

        Q,R,P = linalg.qr(W, mode='full', pivoting = True)
        Perm = np.zeros([len(P),len(P)],dtype=int)
        for w in range(0,len(P)):
            Perm[P[w],w] = 1

        sparse = np.zeros([l_s,3])
        for i in range(l_s):
            sparse[i,0] = Data[P[i],0]
            sparse[i,1] = Data[P[i],1]
            sparse[i,2] = Data[P[i],3]
        
        
        # Selecting Relevant columns of Kernel into B Matrix
        Mat = Gaussian.dot(Perm)
        B = Mat[:,0:l_s];
        
        ## Getting the Permutation matrix for the coordinate vector
        ttx = sparse[:,0]
        ttx_sort = np.argsort(ttx)
        Permmx = np.zeros([len(ttx_sort),len(ttx_sort)],dtype=int)
        for w in range(0,len(ttx_sort)):
            Permmx[w,ttx_sort[w]] = 1
            
        tty = sparse[:,1]
        tty_sort = np.argsort(tty)
        Permmy = np.zeros([len(tty_sort),len(tty_sort)],dtype=int)
        for w in range(0,len(tty_sort)):
            Permmy[w,tty_sort[w]] = 1

        
        ### COmputing the Penalty and Fitting the Regularization network
        q1x,q1y = 1,1
        Pex_temp = Penalty_p(q1x,l_s)
        Pe_x = Permmx.T.dot(Pex_temp).dot(Permmx)
        Pey_temp = Penalty_p(q1y,l_s)
        Pe_y = Permmy.T.dot(Pey_temp).dot(Permmy)
        args = (Data,B,Pe_x,Pe_y)
        bnds = [(1.0e-12, None),(1.0e-12, None)]
        par = [0.1,0.1]
        l = minimize(Cost_func_2D,par,args,bounds=bnds,method='SLSQP')
        lam1_x_1 = l.x[0]
        lam1_y_1 = l.x[1]
        fun1_1 = l.fun
        
        
        q1x,q1y = 1,2
        Pex_temp = Penalty_p(q1x,l_s)
        Pe_x = Permmx.T.dot(Pex_temp).dot(Permmx)
        Pey_temp = Penalty_p(q1y,l_s)
        Pe_y = Permmy.T.dot(Pey_temp).dot(Permmy)
        args = (Data,B,Pe_x,Pe_y)
        bnds = [(1.0e-12, None),(1.0e-12, None)]
        par = [0.1,0.1]
        l = minimize(Cost_func_2D,par,args,bounds=bnds,method='SLSQP')
        lam1_x_2 = l.x[0]
        lam1_y_2 = l.x[1]
        fun1_2 = l.fun    
        
        if fun1_1<=fun1_2:
            fun1 = fun1_1
            lam_x1 = lam1_x_1
            lam_y1 = lam1_y_1
            optq_x1 = 1
            optq_y1 = 1
        else:
            fun1 = fun1_2
            lam_x1 = lam1_x_2
            lam_y1 = lam1_y_2
            optq_x1 = 1
            optq_y1 = 2
    
        
        

        q2x,q2y = 2,1
        Pex_temp = Penalty_p(q2x,l_s)
        Pe_x = Permmx.T.dot(Pex_temp).dot(Permmx)
        Pey_temp = Penalty_p(q2y,l_s)
        Pe_y = Permmy.T.dot(Pey_temp).dot(Permmy)
        args = (Data,B,Pe_x,Pe_y)
        bnds = [(1.0e-12, None),(1.0e-12, None)]
        par = [0.1,0.1]
        l = minimize(Cost_func_2D,par,args,bounds=bnds,method='SLSQP')
        lam2_x_1 = l.x[0]
        lam2_y_1 = l.x[1]
        fun2_1 = l.fun
        
        
        q2x,q2y = 2,2
        Pex_temp = Penalty_p(q2x,l_s)
        Pe_x = Permmx.T.dot(Pex_temp).dot(Permmx)
        Pey_temp = Penalty_p(q2y,l_s)
        Pe_y = Permmy.T.dot(Pey_temp).dot(Permmy)
        args = (Data,B,Pe_x,Pe_y)
        bnds = [(1.0e-12, None),(1.0e-12, None)]
        par = [0.1,0.1]
        l = minimize(Cost_func_2D,par,args,bounds=bnds,method='SLSQP')
        lam2_x_2 = l.x[0]
        lam2_y_2 = l.x[1]
        fun2_2 = l.fun
        
        if fun2_1<=fun2_2:
            fun2 = fun2_1
            lam_x2 = lam2_x_1
            lam_y2 = lam2_y_1
            optq_x2 = 2
            optq_y2 = 1
        else:
            fun2 = fun2_2
            lam_x2 = lam2_x_2
            lam_y2 = lam2_y_2
            optq_x2 = 2
            optq_y2 = 2
            
            
        if fun1 <= fun2:
            fun_vec.append(fun1)
            lam_x = lam_x1
            lam_y = lam_y1
            optq_x = optq_x1
            optq_y = optq_y1
        else:
            fun_vec.append(fun2)
            lam_x = lam_x2
            lam_y = lam_y2
            optq_x = optq_x2
            optq_y = optq_y2
    
    
    
    
        #8. Returning the mean Prediction
        Bpred = np.zeros([Data_pred.shape[0],l_s])
        for i in range(Data_pred.shape[0]):
            for j in range(l_s):
                temp = distance.euclidean(Data_pred[i,:], np.array([sparse[j,0],sparse[j,1]]))
                Bpred[i,j]  = np.exp(-((temp**2)/epsilon))
        
        
        Pex_temp = Penalty_p(optq_x,l_s)
        Pe_x = Permmx.T.dot(Pex_temp).dot(Permmx)
        Pey_temp = Penalty_p(optq_y,l_s)
        Pe_y = Permmy.T.dot(Pey_temp).dot(Permmy)
        inver = np.linalg.inv(B.T.dot(B) + n*lam_x*Pe_x + n*lam_y*Pe_y)
        theta = inver.dot(B.T.dot(Data[:,3].reshape(-1,1)))
        pred = Bpred.dot(theta)
        
        #9. Standard Error bounds
        nr = (Data[:,3].reshape(-1,1) - B.dot(theta)).reshape(-1,1)
        term = B.dot(inver.dot(B.T))
        df_res = n - 2*np.trace(term) + np.trace(term.dot(term.T))
        sigmasq = (nr.T.dot(nr))/(df_res)
        sigmasq = sigmasq[0][0]
        std = np.sqrt(np.diag(sigmasq*Bpred.dot(inver).dot(Bpred.T)))
        stdev_t = sp.stats.t._ppf((1+confidence)/2.,df_res)*std
        results[s] = [pred,stdev_t]
        qoptx.append(optq_x)
        qopty.append(optq_y)
        Sp[s] = sparse
        print(s)
        if l_s == n:
            break
        s = s+1
    return [results,Sp,qoptx,qopty,fun_vec]







