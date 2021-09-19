
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage as nd 
from scipy import signal as sg
from scipy import interpolate as ip
from data_prep_utils.misc import matlab_style_gauss2D, matlab_style_sobel2D
from data_prep_utils.inpaint_nan3 import fill_in_outlier_points
from data_prep_utils.defineParams import defineAllParams
#get parameter function that check the parameter

'''
code adapted from https://github.com/steventan0110/OCT_preprocess
implementation adapted to make it more similar to matlab implementation
'''
def fill_in_zero_values(img_vol):
    '''
    fill in zero values in img_vol by extrapolating
    
    inputs: 
    1. img_vol: shape (height, width, number of scans)
    
    returns: 
    1. temp vol: cleaned up volume
    
    '''
    
    # #handle zero or nan values on the borders
    img_vol[np.isnan(img_vol)] = 0

    # #fill in from the left side:
    inds = np.argmax(img_vol>0, axis = 1) 
    # fill in zero pixels at the edge
    #in matlab the y-axis is not automatically deleted, so here the loop needs to change
    for i in range(img_vol.shape[0]): # loop through pixels in height
        for j in range(img_vol.shape[2]): # loop through scans
            p = inds[i,j]
            if p > 0 and p < i: # second to second last 
                if p < img_vol.shape[1] - 3: # left side 
                    #avoid using low intensity edge pixels
                    # 0:p -> p+2 
                    img_vol[i,:(p+2), j] = img_vol[i,(p+2), j]
                else:
                    img_vol[i,:p, j] = img_vol[i,p,j]


    #fill in from the right side
    temp = np.fliplr(img_vol > 0) #index of last nonzero value
    inds = np.argmax(temp>0, axis = 1)
    inds = img_vol.shape[1] - inds -1 #use -1 instead of + 1 for numpy

    for i in range(img_vol.shape[0]): 
        for j in range(img_vol.shape[2]):
            p = inds[i,j]
            if p < img_vol.shape[1]-1 and img_vol.shape[1] - (p+1) < i+1:
                if p >1:
                    #avoid using low intensity edge pixels
                    img_vol[i, (p-1):, j] = img_vol[i,(p-2), j]
                else:
                    img_vol[i, (p+1):, j] = img_vol[i,p,j]    

    # fill in 0 pixels at the top and bottom
    #fill in from top:
    mv = np.mean(img_vol)

    #same process for inds
    inds = np.argmax(img_vol>0, axis = 0)

    # must remember that python is exclusive, matlab is inclusive. p ind is already changed
    for i in range(img_vol.shape[1]):
        for j in range(img_vol.shape[2]):
            p = inds[i,j]
            if p > 0:
                if  p < img_vol.shape[0] -3:
                    #avoid using low intensity edge pixels
                    if img_vol[p+2,i,j] < mv:
                        img_vol[:(p+2),i,j] = img_vol[p+2,i,j]

                    else:

                        img_vol[:(p+2), i, j]= mv 
                else:

                    img_vol[:p, i, j] = img_vol[p,i,j]

    #fill in from the bottom

    temp_vol = np.flipud(img_vol > 0) #index of last nonzero value
    inds = np.argmax(temp_vol>0, axis = 0)
    inds = img_vol.shape[0] - inds - 1 #use -1 instead of + 1 for numpy

    count=1
    temp_vol = img_vol.copy()
    for i in range(img_vol.shape[1]): # by width
        for j in range(img_vol.shape[2]): # by b-scan
            p = inds[i,j]
            if p < img_vol.shape[0]-1: # need to be not the bottom
                if p > 1:
                    #avoid using low intensity edge pixels
                    temp_vol[(p-1):, i,j] = img_vol[(p-2),i,j]
                else:
                    temp_vol[(p+1):,i,j] = img_vol[p,i,j] 
    return temp_vol


def find_layers(grad, distConst, maxdist, isosThresh, maxdist_bm):
    '''
    find ilm, isos and bm position
    
    input:
    1. grad: gradient image
    2. distConst: distance from largest gradient
    3, maxdist: maximum distance from ILM to ISOS
    4. isosThresh: minimum distance from isos to bm
    5. maxdist_bm: maximum distance from ISOS to BM
    
    returns:
    1. ilm: coordinates of ilm layer, with outliers
    2. isos: coordinates of isos layer, with outliers
    3. bm: coordinates of bm layer, with outliers
    '''
    
    grad_o = grad.copy()
    max1pos = np.argmax(grad, axis =0)

    #to check if max1pos is vector, we have to use the shape of max1pos
    m_size = max1pos.shape
    if m_size[0] == 1 or m_size[1] == 1:
        max1pos =np.transpose(max1pos)
    
    #Find the largest gradient to the max gradient at distance of
    #at least distCount away but not more than maxdist away => set those impossible regions to gradient=0
    for i in range(grad.shape[1]):
        for j in range(grad.shape[2]):
            dc = distConst
            if (max1pos[i,j] - distConst) < 1: # if the largest gradient is near the edge

                dc = max1pos[i,j] -1
            elif (max1pos[i,j] + distConst) > grad.shape[0]: # if it exceeds gradient shape

                dc = grad.shape[0] - max1pos[i,j]

            grad[int(max1pos[i,j]-dc):int(max1pos[i,j]+dc)+1, i,j] = 0 # set all the gradient to 0 
            #max distance => set to 0
            if (max1pos[i,j] - maxdist) > 0:
                grad[:int(max1pos[i,j]-maxdist)+1,i,j] = 0
            if (max1pos[i,j] + maxdist) <= grad.shape[0]:
                grad[int(max1pos[i,j]+maxdist):,i,j] = 0

    max2pos = np.argmax(grad, axis =0)
    m2_size  =max2pos.shape 
    if m2_size[0] == 1 or m2_size[1] == 1:
        max2pos =np.transpose(max2pos)

    # find ilm and isos
    ilm = np.minimum(max1pos, max2pos)
    isos = np.maximum(max1pos, max2pos) 

    #Fill in BM boundary
    grad = grad_o

    #BM is largest negative gradient below the ISOS
    for i in range(grad.shape[1]):
        for j in range(grad.shape[2]):
            grad[:int(isos[i,j]+isosThresh)+1, i ,j] = 0
            if (isos[i,j]+maxdist_bm) <= grad.shape[0]:
                grad[int(isos[i,j]+maxdist_bm):,i,j] = 0

    #To encourage boundary points closer to the top of the image, weight linearly by depth
    isos_temp = (grad.shape[0] - (isos[np.newaxis,:,:]  + maxdist_bm))
    lin = np.transpose(np.arange(grad.shape[0])).reshape(496,1,1) + isos_temp
    lin = -0.5/grad.shape[0] * lin +1
    grad = grad*lin

    bot = np.argmin(grad, axis = 0) #no need to squeeze for python
    bot_sz  = bot.shape
    if bot_sz[0] == 1 or bot_sz[1] == 1:
        print('reach here') #shouldn't reach here with given input
        bot =np.transpose(bot)
    bm  = bot # just the min
    
    return ilm, isos, bm


def retinaDetector(img_vol, header, paramSet):        
        
    newParamSet = defineAllParams(paramSet, header)
    sz, hd, temp, mf_k, bsc_indep = newParamSet['sz'], newParamSet['hd'], newParamSet['temp'], newParamSet['mf_k'], newParamSet['bsc_indep']
    sigma_lat, sigma_ax = newParamSet['sigma_lat'], newParamSet['sigma_ax']
    distConst, maxdist, maxdist_bm = newParamSet['distConst'], newParamSet['maxdist'], newParamSet['maxdist_bm']
    isosThresh, dc_thresh = newParamSet['isosThresh'], newParamSet['dc_thresh']
    sigma_tp_ilm, sigma_tp_isos, sigma_tp_bm, sigma_lat_ilm, sigma_lat_isos, sigma_lat_bm = newParamSet['sigma_tp_ilm'], newParamSet['sigma_tp_isos'], newParamSet['sigma_tp_bm'], newParamSet['sigma_lat_ilm'], newParamSet['sigma_lat_isos'], newParamSet['sigma_lat_bm']

    # #Pre-processing
    temp_vol = fill_in_zero_values(img_vol)

    sigma_ax = float(sigma_ax)
    sigma_lat = float(sigma_lat)
    
    filter1 = matlab_style_gauss2D((2*np.round(2*sigma_ax) + 1,1),sigma_ax)
    filter2 = matlab_style_gauss2D((1,2*np.round(2*sigma_ax) + 1),sigma_lat)

    # filtering the image
    grad = scipy.ndimage.correlate(temp_vol, np.expand_dims(filter1, axis=2), mode='nearest')
    grad = scipy.ndimage.correlate(grad, np.expand_dims(filter2, axis=2), mode='nearest')
    grad = -scipy.ndimage.correlate(grad, np.expand_dims(matlab_style_sobel2D(), axis=2), mode='nearest')
        
    # find layers
    ilm, isos, bm = find_layers(grad, distConst, maxdist, isosThresh, maxdist_bm)
    
    #detect outliers
    if bsc_indep: #not reached in the given data
        th = bm - ilm
        th_med = sg.medfilt2d(th, mf_k.reshape(1,2))
        bpt = (abs(th - th_med) > dc_thresh)
    else:
        mf_k = mf_k.astype(int)
        ilm_med = nd.median_filter(ilm.astype(float), [mf_k[0,0], mf_k[0,1]])
        isos_med = nd.median_filter(isos.astype(float), [mf_k[0,0], mf_k[0,1]])
        bm_med = nd.median_filter(bm.astype(float), [mf_k[0,0], mf_k[0,1]])
        dc_thresh = float(dc_thresh)
        ilmt = np.abs(ilm - ilm_med)
        isost = np.abs(isos - isos_med)
        bmt = np.abs(bm - bm_med)
        par = np.maximum(ilmt, isost)
        par = np.maximum(par, bmt) #the combined maximum of three absolute difference
        bpt = par > dc_thresh


    ilm2, isos2, bm2, nbpt = fill_in_outlier_points(ilm, isos, bm, bpt, bsc_indep)
    #Get final boundaries by smoothing
    #smooth surfaces
    sigma_tp_ilm, sigma_tp_isos, sigma_tp_bm = float(sigma_tp_ilm), float(sigma_tp_isos), float(sigma_tp_bm)
    sigma_lat_ilm, sigma_lat_isos, sigma_lat_bm = float(sigma_lat_ilm), float(sigma_lat_isos), float(sigma_lat_bm)
    ilm3 = ilm2.T
    isos3 = isos2.T
    bm3 = bm2.T
    if not bsc_indep:
        filtera = matlab_style_gauss2D((2*np.round(3*sigma_tp_ilm) + 1,1),sigma_tp_ilm)
        ilm3 = scipy.ndimage.correlate(ilm3, filtera, mode='nearest')

        filterb = matlab_style_gauss2D((2*np.round(3*sigma_tp_isos) + 1,1),sigma_tp_isos)
        isos3 = scipy.ndimage.correlate(isos3, filterb, mode='nearest')

        filterc = matlab_style_gauss2D((2*np.round(3*sigma_tp_bm) + 1,1),sigma_tp_bm)
        bm3 = scipy.ndimage.correlate(bm3, filterc, mode='nearest')

        filterd = matlab_style_gauss2D((2*np.round(3*sigma_lat_bm) + 1,1),sigma_lat_bm)
        bm3 = scipy.ndimage.correlate(bm3, filterd, mode='nearest')

    filtere = matlab_style_gauss2D((2*np.round(3*sigma_lat_ilm) + 1,1),sigma_lat_ilm)
    ilm3 = scipy.ndimage.correlate(ilm3, filtere, mode='nearest')

    filterf = matlab_style_gauss2D((2*np.round(3*sigma_lat_isos) + 1,1),sigma_lat_isos)
    isos3 = scipy.ndimage.correlate(isos3, filterf, mode='nearest')

    #need to transfer all the image to filter function
    #Enforce ordering and a very small minimum thickness

    bmilm = (bm3 -ilm3)*header['ScaleZ']*1000 <100
    ilm3[bmilm] = bm3[bmilm] - 100/header['ScaleZ']/1000
    bmisos = (bm3 -isos3)*header['ScaleZ']*1000 <10
    isos3[bmisos] = bm3[bmisos] - 10/header['ScaleZ']/1000
    isosilm = (isos3-ilm3)*header['ScaleZ']*1000 < 90
    isos3[isosilm] = ilm3[isosilm] + 90/header['ScaleZ']/1000
    
    #Make sure that we are not out of the volume
    ilm3[ilm3 <0] = 1
    ilm3[ilm3> img_vol.shape[0]-1] = img_vol.shape[0]
    isos3[isos3 <0] = 1
    isos3[isos3 > img_vol.shape[0]-1] = img_vol.shape[0]
    bm3[bm3<0] = 1
    bm3[bm3>img_vol.shape[0]-1] = img_vol.shape[0]

    #create mask volume, retina and positional map
    retinaMask = np.zeros(img_vol.shape)
    positional_map = np.zeros(img_vol.shape)
    for i in range(img_vol.shape[1]):
        for j in range(grad.shape[2]):
            retinaMask[int(np.round(ilm3[i,j])):int(np.round(isos3[i,j])), i, j] = 1
            retinaMask[int(np.round(isos3[i,j])):int(np.round(bm3[i,j]))+1, i, j] =1
            I = (np.arange(int(np.round(ilm3[i,j])), int(np.round(bm3[i,j]))+1)-(int(np.round(ilm3[i,j]))-1))/(1+int(np.round(bm3[i,j])) - int(np.round(ilm3[i,j])))
            positional_map[int(np.round(ilm3[i,j])): int(np.round(bm3[i,j]))+1, i, j] = I

    ilm_cat = ilm3.reshape(ilm3.shape[0], ilm3.shape[1], 1)
    isos_cat = isos3.reshape(isos3.shape[0], isos3.shape[1], 1)
    bm_cat = bm3.reshape(bm3.shape[0], bm3.shape[1], 1)

    boundaries = np.concatenate((ilm_cat, isos_cat, bm_cat), axis= 2)
    #define the shift amount here - mean shift per scan
    # DON'T UNDERSTAND WHY NEED TO CALCULATE STEMP-> ISN'T IT JUST HALF OF THE IMAGE
    stemp_bm3 = np.mean(bm3, axis=0)+1 + (np.round(img_vol.shape[0]/2) - np.mean(bm3, axis=0)-1)
    shifts_bm3 = bm3 - stemp_bm3.reshape((1,-1)) # follow matlab
    
    return [retinaMask, positional_map, shifts_bm3, boundaries, nbpt]