'''
adapted from https://github.com/steventan0110/OCT_preprocess
'''
import math
import numpy as np
from astropy.convolution import convolve
from scipy.signal import find_peaks 
from data_prep_utils.misc import build_mask, sp_noise
from data_prep_utils.retinaFlatten import retinaFlatten
from data_prep_utils.retinaDetect import retinaDetector

class preprocessData():
    def __init__(self, img_vol, header, pp_params, probs, scanner_type, annotations, data_list):
        
        '''
        preprocess healthy and MS data
        1.img_vol: Bscans of shape (height, width, bscans)
        2. header: some image information
        3. pp_params: algorithm info
        4. probs: for synthetic maps
        5. scanner_type: default spectralis
        6. annotations: gt annotations (num layers, bscans, height, width)
        7. data_list: what to obtain 
        '''
        self.img_vol = img_vol
        self.header = header
        self.probs = probs
        self.pp_params = pp_params
        self.scanner_type = scanner_type
        self.annotations = annotations
        self.flatten_list = ['data', 'rmask', 'pmap', 'lmap']
        self.height = img_vol.shape[0]
        self.width = img_vol.shape[1]
        self.bscans = img_vol.shape[2]
        self.layers = annotations.shape[2]
        self.smap_boundaries = [6,7]
    
        self.data_store = {k:None for k in data_list}
        self.layers_map = None
        self.retina_mask = None
        self.positional_map = None
        self.intensity_profile = None
        self.synthetic_map = None
        self.image_with_noise = None
        self.shifts = None
        self.boundaries = None
        self.nbpt = None
        
    def preprocess(self):
        
        self.detectRetina()
        self.getGTAnnotations()

        if self.pp_params['flatten'] == True:
            print('Flattening data')
            self.flattenRetina(names = self.flatten_list)
        if self.pp_params['get_smap'] == True:
            print('Preparing S map')
            self.prepareSMap()
        if self.pp_params['remove_bkgd'] == True:        
            print('Removing background')
            self.img_vol = np.multiply(self.img_vol, self.retina_mask)
            if self.pp_params['get_smap'] == True:
                self.synthetic_map = np.multiply(self.synthetic_map,self.retina_mask)
        
        self.image_with_noise = np.multiply(self.image_with_noise,self.retina_mask) # this happens by default to remove overlap noise

        for k, value in self.data_store.items():
            if k == 'data': self.data_store[k] = self.img_vol
            if k == 'pmap': self.data_store[k] = self.positional_map
            if k == 'rmask': self.data_store[k] = self.retina_mask
            if k == 'lmap': self.data_store[k] = self.layers_map
            if k == 'smap' and self.synthetic_map is not None: self.data_store[k] = self.synthetic_map
            if k == 'smap_noise' and self.image_with_noise is not None: self.data_store[k] = self.image_with_noise
        
            
    def detectRetina(self):

        [self.retina_mask, self.positional_map, self.shifts, self.boundaries, self.nbpt] = retinaDetector(self.img_vol, self.header, self.pp_params['retinadetector_type'])

        print("done! {} outlier points".format(self.nbpt))
        self.retina_mask = self.retina_mask > 0;

        if self.nbpt > 0.5*self.img_vol.shape[1]:
            print('Warning: poor fit of retina boundaries detected ({} outlier points). Check for artifacts in the data.\n'.format(self.nbpt))
            
    def getGTAnnotations(self):
            # prepare annotations
        background=np.ones((self.height, self.width, self.bscans))
        layers_map = np.zeros((self.layers, self.height, self.width, self.bscans))
        for scan in range(self.bscans):
            layers_map[:,:,:,scan] = build_mask(self.annotations[:,scan,:],self.height,self.width)
            
        layers_map[self.layers-1]=background-np.sum(layers_map[:-1,:,:,:],0)
        
        self.layers_map = layers_map
        
        
    def constructSMap(self, synthetic_map, intensity_profile):
        '''
        taken from xxx source 
        '''
        
        image_with_noise=np.zeros((self.height, self.width, self.bscans))
        synthetic_map2 = np.copy(synthetic_map)
        
        # constructing synthetic map: get the areas with shadows 
        for scan in range(self.bscans):

            threshold = np.nanmean(np.where(intensity_profile[:,:,scan][:]>0, intensity_profile[:,:,scan][:], np.nan)) - 1.5*np.nanstd(np.nanmean(np.where(intensity_profile[:,:,scan]>0, intensity_profile[:,:,scan], np.nan), axis = 0))
            average_intensity = convolve(np.nanmean(np.where(intensity_profile[:,:,scan]>0, intensity_profile[:,:,scan], np.nan),axis=0), np.ones(9)/9, nan_treatment='interpolate') # average filtering
            troughs, props = find_peaks(-average_intensity, -threshold, width=5)

            if len(troughs)>0:
                for idx,trough in enumerate(troughs):
                    synthetic_map2[:,math.floor(props['left_ips'][idx]):math.ceil(props['right_ips'][idx]),scan] = self.img_vol[:,math.floor(props['left_ips'][idx]):math.ceil(props['right_ips'][idx]),scan]
            image_with_noise[:,:,scan] = sp_noise(synthetic_map2[:,:,scan], self.probs)

        
        
        return synthetic_map, image_with_noise
    
    
    def prepareSMap(self):
        synthetic_map=np.zeros((self.height, self.width, self.bscans))
        intensity_profile=np.zeros((self.height, self.width, self.bscans))
        
        
        for layer in range(self.layers-1):
            for scan in range(self.bscans):
                label = np.copy(self.layers_map[layer,:,:,scan]) # per layer and per scan
                label = (label==1)
                img_vol_scan = np.copy(self.img_vol[:,:,scan]) # per image
                img_vol_scan[~label] = 0 # isolate the layer
                # creating synthetic map
                synthetic_map[:,:,scan][label] = np.nanmean(np.where(img_vol_scan>0, img_vol_scan, np.nan))
                # for the last two layers, add to intensity profile
                if layer in self.smap_boundaries:
                    intensity_profile[:,:,scan] += img_vol_scan
                    
        self.synthetic_map, self.image_with_noise = self.constructSMap(synthetic_map, intensity_profile)
        
    def flattenRetina(self, names):
        
        bds = self.boundaries
        shifts = self.shifts
        tb = bds[:,:,0] - shifts

        if np.any(tb <0): #For the example case, won't get in
            shifts = shifts + np.amin(tb)
            #center
            d = np.amin(self.height - (bds[:,:,-1] - shifts))
            shifts = shifts - np.amin(d)/2
            
        if 'data' in names: self.img_vol = retinaFlatten(self.img_vol, shifts, 'linear')
        if 'lmap' in names:
            background=np.ones((self.height, self.width, self.bscans))
            # prepare intensity profile for synthetic map
            for layer in range(self.layers-1): # except background
                # flatten the layer map
                self.layers_map[layer,:,:,:] = retinaFlatten(self.layers_map[layer,:,:,:], shifts, 'nearest') # layer mask
            # get background
            self.layers_map[self.layers-1]=background-np.sum(self.layers_map[:-1,:,:,:],0)
            
        if 'pmap' in names: self.positional_map = retinaFlatten(self.positional_map, shifts, 'nearest')
        if 'rmask' in names: self.retina_mask = retinaFlatten(self.retina_mask, shifts, 'nearest')
