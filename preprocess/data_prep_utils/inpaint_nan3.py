import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.io import loadmat
from scipy import interpolate as ip
from scipy.interpolate import CubicSpline
from scipy import sparse as sps
import numpy.linalg as lin
import scipy.ndimage
import pdb


def identify_neighbors(n, m, nan_list, talks_to):
    if len(nan_list)>0:
        nan_count = nan_list.shape[0]
        talk_count = talks_to.shape[0]
        nn = np.zeros((nan_count*talk_count,2))
        j = np.array([0,nan_count]);
        for i in range(talk_count):
            nn[j[0]:j[1],:] = nan_list[:,1:3]+np.tile(talks_to[i,:],(nan_count,1))
            j+=nan_count
        
            
        L = (nn[:,0]<0) | (nn[:,0]>n-1) | (nn[:,1]<0) | (nn[:,1]>m-1)
        nn = nn[~L]
        
        neighbors_list = np.ravel_multi_index([nn[:,0].astype(int),nn[:,1].astype(int)], [n,m], order='F')
        neighbors_list = np.concatenate((np.expand_dims(neighbors_list,axis=1), nn), axis=1)
        neighbors_list = np.unique(neighbors_list, axis=0).astype(int)

        neighbors_list2 =[]
        for i in neighbors_list.tolist():
            if i not in nan_list.tolist():
                neighbors_list2.append(i)
        return neighbors_list2
    else:
        neighbors_list=[];
    

def inpaint_nans(image):
    '''
    impaint nans by solving sparse arrays
    previously taken from https://www.mathworks.com/matlabcentral/fileexchange/4551-inpaint_nans
    '''
    n, m = image.shape[0], image.shape[1]
    nm = n*m
    reshaped_image = np.reshape(image.T,-1)
    
    k = np.isnan(reshaped_image)
    nan_list = np.array(np.where(k))[0]
    (known_list) = np.array(np.where(~np.isnan(reshaped_image)))[0]
    nan_count = len(nan_list)
    nc,nr = np.unravel_index(nan_list, (m,n))
    nan_list = np.concatenate((np.expand_dims(nan_list,axis=1), np.expand_dims(nr,axis=1), np.expand_dims(nc,axis=1)), axis=1)
    
    talks_to = np.array([[-1,0],[0,-1],[1,0],[0,1]])
    
    neighbors_list=identify_neighbors(n,m,nan_list,talks_to)
    
    all_list=np.concatenate((nan_list,neighbors_list), axis=0)
    
    L = np.array(np.where((all_list[:,1] > 0) & (all_list[:,1] < n-1)))[0]
    nl = len(L)
    if nl>0:
        a = np.reshape(np.tile(np.expand_dims([1,-2,1],axis=0),(nl,1)),-1)
        b = np.reshape(np.tile(np.expand_dims(all_list[L,0],axis=1),(1,3)),-1)
        c = np.reshape((np.tile(np.expand_dims(all_list[L,0],axis=1),(1,3))+np.tile(np.expand_dims([-1,0,1],axis=0),(nl,1))),-1)
        fda = sps.csr_matrix((a, (b,c)), shape=(nm, nm), dtype=float)
#         fda.eliminate_zeros()
#         fda.todense()
    else: ## to fill up
        fda = sps.csr_matrix(shape=(m,n),dtype=float)
    L = np.array(np.where((all_list[:,2] > 0) & (all_list[:,2] < m-1)))[0]
    nl = len(L)
    if nl>0:  
        a = np.reshape(np.tile(np.expand_dims([1,-2,1],axis=0),(nl,1)),-1)
        b = np.reshape(np.tile(np.expand_dims(all_list[L,0],axis=1),(1,3)),-1)
        c = np.reshape((np.tile(np.expand_dims(all_list[L,0],axis=1),(1,3))+np.tile(np.expand_dims([-n,0,n],axis=0),(nl,1))),-1)
        fda = fda + sps.csr_matrix((a, (b,c)), shape=(nm, nm), dtype=float)
    
#         fda.eliminate_zeros()
#         fda.todense()

    rhs= -fda[:,known_list]*reshaped_image[known_list] # give weight to the elemnts
    
    # nan_list[:,0] is equal, rhs should have no problem
    (r, k) =np.any(fda[:,nan_list[:,0]]!=0,axis=0).nonzero() #problem with this line
    r = np.unique(r)
    B=reshaped_image
    fda2 = fda[r,:]
    
    x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = sps.linalg.lsqr(fda2[:,nan_list[:,0]], rhs[r])
    B[nan_list[:,0]] = x
    B=np.reshape(B,(m,n))    
    return B

def fill_in_outlier_points(ilm, isos, bm, bpt, bsc_indep):
    '''
    fill in outlier points with impaint_nan
    '''
    
    #Fill in outlier points:
    ilm = ilm.astype(float)
    isos = isos.astype(float)
    bm = bm.astype(float)
    ilm[bpt] = np.nan  #find correspondance of nan
    isos[bpt] = np.nan
    bm[bpt] = np.nan
    nbpt = 0
    
    if np.any(np.any(bpt)): #since bpt is 2-D
        nbpt = np.sum(bpt)
        if bsc_indep: #not reached, not fully implemented
            x = np.transpose(range(ilm.shape[0]))
            for j in range(ilm.shape[1]):

                #p = polyfit polyfit function
                p = np.polyfit(x[np.invert(bpt[:,j])],bm[np.invert(bpt[:,j]),j],2) #HERE
                bm[:,j] = np.polyval(p,x) # function need to be transfered #HERE
    #                 quit()
            #linearly interpolate ILM and ISOS
            nv = any(np.isnan(ilm))
            xpts = np.arange(ilm.shape[0])
            for j in range(ilm.shape[1]):
                if nv[j]:
                    print('what')
                    nv2 = not np.isnan(ilm[:,j])
                    f = ip.interp1d(xpts[nv2], ilm[nv2, j], xpts, kind='linear', fill_value='extrap')  #HERE
                    ilm[:,j] = f(xpts)
                    f2 = ip.interp1d(xpts[nv2], isos[nv2, j], xpts, kind='linear', fill_value='extrap')  #HERE
                    isos[:,j] = f2(xpts)
                    #bm = interp1
        else:
            #temporary replacement of the inpaint_nan function
            ilm2 = inpaint_nans(ilm)
            isos2 = inpaint_nans(isos)
            bm2 = inpaint_nans(bm) 
            
    return ilm2, isos2, bm2, nbpt
