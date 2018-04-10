import numpy as np

import cv2
import imutils
import sys
def combine(image,image2):
               a,b=image2.shape[:2]
               image=cv2.resize(image,(b,a),interpolation = cv2.INTER_LINEAR)
               imgs = [image, image2]
               imgs_comb = np.vstack((np.asarray(i) for i in imgs))
               return imgs_comb
def preprocessing(image):
    orig=image.copy
    shape=image.shape
    w=shape[1]
    h=shape[0]
    
    #extension of the image
    #=============================================================================
    base_size=h+10,w+10,3
    base=np.zeros(base_size,dtype=np.uint8)
    base[5:h+5,5:w+5]=image 
    orig = base.copy()
    image=base
    
    #median filter
    #=============================================================================
    image=cv2.medianBlur(image,51)
    cv2.imwrite("stage1.jpg",image)
    
    #colourtogray
    #=============================================================================
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imwrite("stage2.jpg",image)
    
    #cannyedgedetector
    #=============================================================================
    edged = cv2.Canny(image, 75, 200)
    cv2.imwrite("stage3.jpg",edged)
    
    #find contours
    #=============================================================================
    cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1] 
    cv2.drawContours(orig,cnts, -1, (0,255,0), 3)
    cv2.imwrite("stage4.jpg",orig)
    
    #approximate a polygon
    #=============================================================================
    maxi=0
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    for c in cnts:
        maxi=max(maxi, cv2.arcLength(c,False))
        if(maxi==cv2.arcLength(c,False)):
            ScreenCnt=c
    cv2.drawContours(orig,[ScreenCnt], -1, (0, 255, 0), 2)
    epsilon = 0.01*cv2.arcLength(ScreenCnt,False)
    approx = cv2.approxPolyDP(ScreenCnt,epsilon,False)
    cv2.drawContours(base,[approx], -1, (0,255,0), 3)
    cv2.imwrite("stage5.jpg",base)
    #=============================================================================
    #Four corners of a image
    min_ht=sys.maxint
    max_ht=0
    min_wt=sys.maxint
    max_wt=0
    try:
        
         for a in approx:
             min_ht=min(a[0][0],min_ht)
             max_ht=max(a[0][0],max_ht)
             min_wt=min(a[0][1],min_wt)
             max_wt=max(a[0][1],max_wt)
         new_image=base[min_wt:max_wt,min_ht:max_ht]
         cv2.imwrite("stage6.jpg",new_image)
         return new_image
         
    # =============================================================================
    except:
        return orig
        
#Getting the image from the dataset.Feeding the image along with the path

image = cv2.imread("yourfile.jpg")
image3=preprocessing(image)



