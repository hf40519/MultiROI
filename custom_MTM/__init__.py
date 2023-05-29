# --------------------------------------------------------------------- #
#                 No Need to Install MTM and skimage                    #
#               Here are both packages implementation                   #
# --------------------------------------------------------------------- #

# from skimage.feature import peak_local_max
# import MTM import NMS
from .NMS import NMS
from .skimage_feature import peak_local_max
import numpy as np
import cv2, warnings, os
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor, as_completed

__all__ = ['NMS']

def computeScoreMap(template, image, method=cv2.TM_CCOEFF_NORMED, mask=None):
    # Convert images if not both 8-bit (OpenCV matchTemplate is only defined for 8-bit OR 32-bit)
    if not (template.dtype == "uint8" and image.dtype == "uint8"):
        template = np.float32(template)
        image = np.float32(image)
        if mask is not None: 
            mask = np.float32(mask)

    if mask is not None:
        if method not in (0, 3):
           mask = None
           warnings.warn("Template matching method not compatible with use of mask (only 0/TM_SQDIFF or 3/TM_CCORR_NORMED).\n-> Ignoring mask.")
        else: # correct method
           # Check that mask has the same dimensions and type than template
            sameDimension = mask.shape == template.shape
            sameType = mask.dtype == template.dtype
            if not (sameDimension and sameType):
                mask = None
                warnings.warn("Mask does not have the same dimension or bit depth than the template.\n-> Ignoring mask.")
    
    # Compute correlation map
    return cv2.matchTemplate(image, template, method, mask=mask)

def _findLocalMax_(corrMap, score_threshold=0.6):
    """Get coordinates of the local maximas with values above a threshold in the image of the correlation map."""
    # IF depending on the shape of the correlation map
    if corrMap.shape == (1, 1): ## Template size = Image size -> Correlation map is a single digit')
        if corrMap[0, 0] >= score_threshold:
            peaks = np.array([[0, 0]])
        else:
            peaks = []
    # use scipy findpeaks for the 1D cases (would allow to specify the relative threshold for the score directly here rather than in the NMS
    elif corrMap.shape[0] == 1:     ## Template is as high as the image, the correlation map is a 1D-array
        peaks = find_peaks(corrMap[0], height=score_threshold) # corrMap[0] to have a proper 1D-array
        peaks = [[0, i] for i in peaks[0]] # 0,i since one coordinate is fixed (the one for which Template = Image)

    elif corrMap.shape[1] == 1: ## Template is as wide as the image, the correlation map is a 1D-array
        peaks = find_peaks(corrMap[:, 0], height=score_threshold)
        peaks = [[i, 0] for i in peaks[0]]
    
    else: # Correlation map is 2D
        peaks = peak_local_max(corrMap, threshold_abs=score_threshold, exclude_border=False).tolist()

    return peaks

def _findLocalMin_(corrMap, score_threshold=0.4):
    """Find coordinates of local minimas with values below a threshold in the image of the correlation map."""
    return _findLocalMax_(-corrMap, -score_threshold)

def _multi_compute(tempTuple, image, method, N_object, score_threshold, xOffset, yOffset, listHit):
    templateName, template = tempTuple[:2]
    mask = None

    if len(tempTuple) >= 3: # ie a mask is also provided
        if method in (0, 3):
            mask = tempTuple[2]
        else:
            warnings.warn("Template matching method not supporting the use of Mask. Use 0/TM_SQDIFF or 3/TM_CCORR_NORMED.")
    
    corrMap = computeScoreMap(template, image, method, mask=mask)

    ## Find possible location of the object
    if N_object == 1: # Detect global Min/Max
        _, _, minLoc, maxLoc = cv2.minMaxLoc(corrMap)
        if method in (0, 1):
            peaks = [minLoc[::-1]] # opposite sorting than in the multiple detection
        else:
            peaks = [maxLoc[::-1]]
    else:# Detect local max or min
        if method in (0,1): # Difference => look for local minima
            peaks = _findLocalMin_(corrMap, score_threshold)
        else:
            peaks = _findLocalMax_(corrMap, score_threshold)

    # Once every peak was detected for this given template
    ## Create a dictionnary for each hit with {'TemplateName':, 'BBox': (x,y,Width, Height), 'Score':coeff}
    height, width = template.shape[:2] # slicing make sure it works for RGB too
    for peak in peaks :
        # append to list of potential hit before Non maxima suppression
        # no need to lock the list, append is thread-safe
        listHit.append({'TemplateName': templateName, 'BBox': (int(peak[1])+xOffset, int(peak[0])+yOffset, width, height), 'Score': corrMap[tuple(peak)]})

def findMatches(listTemplates, image, method=cv2.TM_CCOEFF_NORMED, N_object=float("inf"), score_threshold=0.5, searchBox=None):
    if N_object!=float("inf") and type(N_object) != int:
        raise TypeError("N_object must be an integer")
    elif N_object<1:
        raise ValueError("At least one object should be expected in the image")
    ## Crop image to search region if provided
    if searchBox != None: 
        xOffset, yOffset, searchWidth, searchHeight = searchBox
        image = image[yOffset:yOffset+searchHeight, xOffset:xOffset+searchWidth]
    else:
        xOffset=yOffset=0
    ## 16-bit image are converted to 32-bit for matchTemplate
    if image.dtype == 'uint16': 
        image = np.float32(image)   
    listHit = []
    for templateName, template in listTemplates:
        ## 16-bit image are converted to 32-bit for matchTemplate
        if template.dtype == 'uint16': 
            template = np.float32(template)        
        ## Compute correlation map
        corrMap = cv2.matchTemplate(template, image, method)
        ## Find possible location of the object 
        if N_object == 1: # Detect global Min/Max
            _, _, minLoc, maxLoc = cv2.minMaxLoc(corrMap)
            if method == 1:
                Peaks = [minLoc[::-1]] # opposite sorting than in the multiple detection
            elif method in (3, 5):
                Peaks = [maxLoc[::-1]]
        else: # Detect local max or min
            if method == 1: # Difference => look for local minima
                Peaks = _findLocalMin_(corrMap, score_threshold)
            elif method in (3, 5):
                Peaks = _findLocalMax_(corrMap, score_threshold)
        
        # Once every peak was detected for this given template
        ## Create a dictionnary for each hit with {'TemplateName':, 'BBox': (x,y,Width, Height), 'Score':coeff}
        height, width = template.shape[0:2] # slicing make sure it works for RGB too
        for peak in Peaks:
            coeff  = corrMap[tuple(peak)]
            newHit = {'TemplateName':templateName, 'BBox': [int(peak[1])+xOffset, int(peak[0])+yOffset, width, height], 'Score':coeff}
            # append to list of potential hit before Non maxima suppression
            listHit.append(newHit)
    
    return listHit # All possible hit before Non-Maxima Supression

def custom_matchTemplates(listTemplates, image, method=cv2.TM_CCOEFF_NORMED, N_object=float("inf"), score_threshold=0.5, maxOverlap=0.25, searchBox=None):
    '''
    Search each template in the image, and return the best N_object location which offer the best score and which do not overlap
    Parameters
    ----------
    - listTemplates : list of tuples (LabelString, Grayscale or RGB numpy array)
                    templates to search in each image, associated to a label 
    - image  : Grayscale or RGB numpy array
               image in which to perform the search, it should be the same bitDepth and number of channels than the templates
    - method : int 
                one of OpenCV template matching method (0 to 5), default 5=0-mean cross-correlation
    - N_object: int
                expected number of objects in the image
    - score_threshold: float in range [0,1]
                if N>1, returns local minima/maxima respectively below/above the score_threshold
    - maxOverlap: float in range [0,1]
                This is the maximal value for the ratio of the Intersection Over Union (IoU) area between a pair of bounding boxes.
                If the ratio is over the maxOverlap, the lower score bounding box is discarded.
    - searchBox : tuple (X, Y, Width, Height) in pixel unit
                optional rectangular search region as a tuple
    
    Returns
    -------
    - bestHits: list of match as dictionaries {"TemplateName":string, "BBox":(X, Y, Width, Height), "Score":float}
                if N=1, return the best matches independently of the score_threshold
                if N<inf, returns up to N best matches that passed the score_threshold
                if N=inf, returns all matches that passed the score_threshold
    '''
    if len(listTemplates) == 1: # one class ROI
        listHit = findMatches(listTemplates, image, method, N_object, score_threshold, searchBox)
    else: # Multi class ROI
        ## Crop image to search region if provided
        if searchBox is not None:
            xOffset, yOffset, searchWidth, searchHeight = searchBox
            image = image[yOffset: yOffset+searchHeight, xOffset: xOffset+searchWidth]
        else:
            xOffset = yOffset = 0
        listHit = []
        ## Use multi-threading to iterate through all templates, using half the number of cpu cores available.
        with ThreadPoolExecutor(max_workers=round(os.cpu_count()*.5)) as executor:
            futures = [executor.submit(_multi_compute, tempTuple, image, method, N_object, score_threshold, xOffset, yOffset, listHit) for tempTuple in listTemplates]
            for future in as_completed(futures):
                _ = future.result()
            
    if method == 1:       
        bestHits = NMS(listHit, N_object=N_object, maxOverlap=maxOverlap, sortDescending=False)
    elif method in (3, 5): 
        bestHits = NMS(listHit, N_object=N_object, maxOverlap=maxOverlap, sortDescending=True)
    
    return bestHits