import numpy as np

                   
def defineAllParams(paramSet, header):
    '''
    define all parameters for retinal detection
    '''
    
    def getParams(paramSet):
        params = dict()
        for x in ['default','spectralis','hc','mme']:
            if paramSet == x: 
                #Originally for the spectralis
                params['sigma_lat'] = 16.67
                params['sigma_ax'] = 11.6
                params['distconst'] = 96.68
                params['sigma_lat_ilm'] = 55.56
                params['sigma_lat_isos'] = 55.56
                params['sigma_lat_bm'] = 111.13
                params['maxdist'] = 386.73 # ~100 pixels in spectralis
                params['bsc_indep'] = False
                return params
        if paramSet == 'dme':
            params['sigma_lat'] = 16.67
            params['sigma_ax'] = 11.6
            params['distconst'] = 96.68
            params['sigma_lat_ilm'] = 55.56
            params['sigma_lat_isos'] =55.56
            params['sigma_lat_bm'] = 111.13
            params['maxdist'] = 386.73 # ~100 pixels in spectralis
            params['bsc_indep'] = False
        elif paramSet == 'cirrus':
            params['sigma_lat'] = 2*16.67
            params['sigma_ax'] = 0.5*11.6
            params['distconst'] = 96.68
            params['sigma_lat_ilm'] = 55.56
            params['sigma_lat_isos'] =55.56
            params['sigma_lat_bm'] = 111.13
            params['maxdist'] = 386.73 # ~100 pixels in spectralis
            params['bsc_indep'] = True
        elif paramSet == 'cirrus_sm':
            params['sigma_lat'] = 2*16.67
            params['sigma_ax'] = 0.5*11.6
            params['distconst'] = 96.68
            params['sigma_lat_ilm'] = 55.56
            params['sigma_lat_isos'] =55.56
            params['sigma_lat_bm'] = 200
            params['maxdist'] = 386.73 # ~100 pixels in spectralis
            params['bsc_indep'] = True
        elif paramSet == 'rp':
            params['sigma_lat'] = 2*16.67
            params['sigma_ax'] = 0.5*11.6
            params['distconst'] = 50
            params['sigma_lat_ilm'] = 200
            params['sigma_lat_isos'] = 300
            params['sigma_lat_bm'] = 200
            params['maxdist'] = 386.73 # ~100 pixels in spectralis
            params['bsc_indep'] = True
        elif paramSet == 'phantom':
            params['sigma_lat'] = 5
            params['sigma_ax'] = 5
            params['distconst'] = 150
            params['sigma_lat_ilm'] = 55.56
            params['sigma_lat_isos'] =55.56
            params['sigma_lat_bm'] = 111.13
            params['maxdist'] = 550 # ~100 pixels in spectralis
            params['bsc_indep'] = False
        else:
            print('wrong parameter\n')
        return params
    
    params = getParams(paramSet) #params is a dictionary here

    #maximum distance from ILM to ISOS:
    maxdist = params['maxdist']
    #maximum distance from ISOS to BM:
    maxdist_bm = 116.02
    #Minimum distance from ISOS to BM:
    isosThresh = 20
    #Median filter outlier threshold distance and kernel
    dc_thresh = 10
    mf_k = 140


    #Process B-scans independently
    bsc_indep = params['bsc_indep']

    if 'angle' in header:
        if abs(abs(header['angle'])-90) < 25:
            bsc_indep = 1

    #sigma values for smoothing final surfaces
    sigma_tp_ilm = 91.62
    sigma_tp_isos = 91.62
    sigma_tp_bm = 244.32
    #lateral direction
    sigma_lat_ilm = params['sigma_lat_ilm']
    sigma_lat_isos = params['sigma_lat_isos']
    sigma_lat_bm = params['sigma_lat_bm']

    #convert all values frmo micron to pixel
    sz = header['ScaleZ']*1000
    hd = header['Distance']*1000
    sigma_lat = params['sigma_lat']/(header['ScaleX']*1000)
    sigma_ax = params['sigma_ax']/sz
    distConst = np.round(params['distconst']/sz)
    maxdist = np.round(maxdist/sz)
    maxdist_bm = np.round(maxdist_bm/sz)
    isosThresh = np.round(isosThresh/sz)
    dc_thresh = np.round(dc_thresh/sz*(128/6)*header['Distance'])

    temp = np.round(np.array([(mf_k/(header['ScaleX']*1000)),(mf_k/(header['Distance']*1000))]))
    mf_k = (temp*2 +1).reshape((1,2))
    sigma_tp_ilm = sigma_tp_ilm/hd
    sigma_tp_isos = sigma_tp_isos/hd
    sigma_tp_bm = sigma_tp_bm/hd
    sigma_lat_ilm = sigma_lat_ilm/(header['ScaleX']*1000)
    sigma_lat_isos = sigma_lat_isos/(header['ScaleX']*1000)
    sigma_lat_bm = sigma_lat_bm/(header['ScaleX']*1000)
    
    newParamSet = {}
    newParamSet['sz'], newParamSet['hd'], newParamSet['temp'], newParamSet['mf_k'], newParamSet['bsc_indep'] = sz, hd, temp, mf_k, bsc_indep
    newParamSet['sigma_lat'], newParamSet['sigma_ax'] = sigma_lat, sigma_ax
    newParamSet['distConst'], newParamSet['maxdist'], newParamSet['maxdist_bm'] = distConst, maxdist, maxdist_bm
    newParamSet['isosThresh'], newParamSet['dc_thresh'] = isosThresh, dc_thresh
    newParamSet['sigma_tp_ilm'], newParamSet['sigma_tp_isos'], newParamSet['sigma_tp_bm'], newParamSet['sigma_lat_ilm'], newParamSet['sigma_lat_isos'], newParamSet['sigma_lat_bm'] = sigma_tp_ilm, sigma_tp_isos, sigma_tp_bm, sigma_lat_ilm, sigma_lat_isos, sigma_lat_bm
    
    return newParamSet
