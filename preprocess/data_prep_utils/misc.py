from PIL import Image, ImageFilter
import numpy as np

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    implementation: https://gist.github.com/lucaswiman/1e877a164a69f78694f845eab45c381a
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0.0
        white = 1.0          
#     else:
#         colorspace = image.shape[2]
#         if colorspace == 3:  # RGB
#             black = np.array([0, 0, 0], dtype='uint8')
#             white = np.array([255, 255, 255], dtype='uint8')
#         else:  # RGBA
#             black = np.array([0, 0, 0, 255], dtype='uint8')
#             white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
   
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

def build_mask(annotations,HEIGHT,WIDTH): 

    """ Build mask for 1 scan
    """
    layers=np.zeros((annotations.shape[1],HEIGHT,WIDTH))
    background=np.ones((HEIGHT,WIDTH))
    for layer in range(annotations.shape[1]-1): # loop from 0 to 7
        for i in range(annotations.shape[0]): # loop through the width of the image
            # check that both layers have values>0 at that region. 
            
            # need to -1 because matlab, check if both layers are greater than -1
            if int(round(annotations[i,layer]))-1>0 and int(round(annotations[i,layer+1]))-1>0:
            # build channels of mask
                layers[layer,int(round(annotations[i,layer]))-1:int(round(annotations[i,layer+1]))-1,i]=1 # need to -1 to convert matlab index to python index 
    # get background
    # size of layers: 9,496,1024
    return layers

    
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def matlab_style_sobel2D():
    """
    2D sobel mask - should give the same result as MATLAB's sobel
    """
    h = [[1,2,1],[0,0,0],[-1,-2,-1]];
    return h
    
    
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text) :
    """ Sort files
    """
    return [atoi(c) for c in re.split('(\d+)', text)]

def drawContour(m,s,c,RGB):
    """Draw edges of contour 'c' from segmented image 's' onto 'm' in colour 'RGB'"""
    # Fill contour "c" with white, make all else black
    thisContour = s.point(lambda p:p==c and 255)
    # DEBUG: thisContour.save(f"interim{c}.png")

    # Find edges of this contour and make into Numpy array
    thisEdges   = thisContour.filter(ImageFilter.FIND_EDGES)
    thisEdgesN  = np.array(thisEdges)

    # Paint locations of found edges in color "RGB" onto "main"
    m[np.nonzero(thisEdgesN)] = RGB
    return m


# plt.imshow(seg,cmap="gray")

def drawContourMain(image, layer_map):
    '''
    plot sample image and it's corresponding contour
    
    inputs: 
    image of shape (height, width)
    lmap of shape (number of layers, height, width)
    
    output:
    figure of image with contour overlaid on it
    
    '''

    mainN = np.array(Image.fromarray(np.copy(image)*255).convert(mode="L").convert(mode="RGB"))
    plt.figure(figsize=(20,15))
    plt.imshow(mainN,cmap="gray")

    for layer in range(layer_map.shape[0]):
    # # Load segmented image as greyscale
        seg = Image.fromarray(layer_map[layer].astype(np.float32)).convert(mode="L")
        mainN = drawContour(mainN,seg,1,(25*layer,0,0))   # draw contour 1 in red
    plt.imshow(mainN)