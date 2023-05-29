import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from MTM import matchTemplates
from custom_MTM import custom_matchTemplates

if __name__ == "__main__":
    t1 = cv2.cvtColor(cv2.imread('./poker/t1.jpeg'), cv2.COLOR_BGR2GRAY)
    t2 = cv2.cvtColor(cv2.imread('./poker/t2.jpeg'), cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(cv2.imread('./poker/image.jpeg'), cv2.COLOR_BGR2GRAY)
    
    ratio = 1
    h, w = img.shape[:2]
    h = int(h/ratio)
    w = int(w/ratio)
    img = cv2.resize(img, (w, h))
    h, w = t1.shape[:2]
    h = int(h/ratio)
    w = int(w/ratio)
    t1 = cv2.resize(t1, (w, h))
    h, w = t2.shape[:2]
    h = int(h/ratio)
    w = int(w/ratio)
    t2 = cv2.resize(t2, (w, h))
    
    listTemplate = [("t1", t1), ("t2", t2), ("t1_1", t1), ("t1_2", t1), ("t1_3", t1), ("t1_4", t1)]
    
    m1_time = []
    m2_time = []
    m3_time = []
    
    for _ in range(100):
        start_time = time.time()
        _ = matchTemplates(listTemplate, img, score_threshold=0.8, 
                           # searchBox=(0, 0, 3000, 750),
                        #    N_object=5,
                           method=cv2.TM_CCOEFF_NORMED, maxOverlap=0.1)
        m1_time.append(time.time()-start_time)

        start_time = time.time()
        hits = custom_matchTemplates(listTemplate, img, score_threshold=0.8,
                                     # searchBox=(0, 0, 3000, 750),
                                    #  N_object=5,
                                     method=cv2.TM_CCOEFF_NORMED, maxOverlap=0.1)
        m2_time.append(time.time()-start_time)
        
        start_time = time.time()
        result = cv2.matchTemplate(img, t1, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        m3_time.append(time.time()-start_time)
        
    print(f"MPM: {np.mean(m1_time)}, customMPM: {np.mean(m2_time)}, CV: {np.mean(m3_time)}")
    
    from MTM import drawBoxesOnRGB
    overlay = drawBoxesOnRGB(img, hits, showLabel=True, labelColor=(255, 0, 0), boxColor=(0, 0, 255), labelScale=1,
                             boxThickness=3)
                                               
    plt.imshow(overlay)
    plt.show()