import cv2
import sys
import numpy as np
args = sys.argv
def colorInk(src, dst,ch1Lower, ch1Upper,ch2Lower, ch2Upper,ch3Lower, ch3Upper):
    src = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    lower = [0,0,0]
    upper = [0,0,0]
    lower[0] = ch1Lower
    lower[1] = ch2Lower
    lower[2] = ch3Lower
    upper[0] = ch1Upper
    upper[1] = ch2Upper
    upper[2] = ch3Upper
    hsv = [0,0,0]
    size = src.shape
    tmp = src
    for y in range(0,size[0]):
        for x in range(0,size[1]):
            hsv[0] = src[y,x][0]
            hsv[1] = src[y,x][1]
            hsv[2] = src[y,x][2]

            if lower[0] <= upper[0]:
                if lower[0] <= hsv[0] and hsv[0] <= upper[0] and lower[1] <= hsv[1] and hsv[1] <= upper[1] and lower[2] <= hsv[2] and hsv[2] <= upper[2]:
                    src[y,x][0] = src[y,x][0]
                    src[y,x][1] = src[y,x][1]
                    src[y,x][2] = src[y,x][2]
                    
                else:
                    src[y,x][0] = 0
                    src[y,x][1] = 0
                    src[y,x][2] = 0
            else:
                if lower[0] <= hsv[0] or hsv[0] <= upper[0]:
                    if lower[1] <= hsv[1] and hsv[1] <= upper[1] and lower[2] <= hsv[2] and hsv[2] <= upper[2]:
                        src[y,x][0] = src[y,x][0]
                        src[y,x][1] = src[y,x][1]
                        src[y,x][2] = src[y,x][2]
                    
                else:
                    src[y,x][0] = 0
                    src[y,x][1] = 0
                    src[y,x][2] = 255

    src = cv2.cvtColor(src,cv2.COLOR_HSV2BGR)
    cv2.imshow('OpenCV', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return src

def colorExtraction1(src, dst,
					 ch1Lower, ch1Upper,
					 ch2Lower, ch2Upper,
					 ch3Lower, ch3Upper):
    src = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
   
    lower = [0,0,0]
    upper = [0,0,0]
    TEKIOU = 0
    akazu = 0
    bkazu = 0
    lower[0] = ch1Lower
    lower[1] = ch2Lower
    lower[2] = ch3Lower
    upper[0] = ch1Upper
    upper[1] = ch2Upper
    upper[2] = ch3Upper
    hsv = [0,0,0]
    size = src.shape
    tmp = np.zeros([size[0], size[1]])
    for y in range(0,size[0]):
        for x in range(0,size[1]):
            hsv[0] = src[y,x][0]
            hsv[1] = src[y,x][1]
            hsv[2] = src[y,x][2]

            if lower[0] <= upper[0]:
                if lower[0] <= hsv[0] and hsv[0] <= upper[0] and lower[1] <= hsv[1] and hsv[1] <= upper[1] and lower[2] <= hsv[2] and hsv[2] <= upper[2]:
                    tmp[y,x]= 255
                    akazu = abs(hsv[0] - 90)
                    bkazu = bkazu + 1
                    TEKIOU = TEKIOU + akazu
                else:
                    tmp[y,x]= 0
            else:
                if lower[0] <= hsv[0] or hsv[0] <= upper[0]:
                    if lower[1] <= hsv[1] and hsv[1] <= upper[1] and lower[2] <= hsv[2] and hsv[2] <= upper[2]:
                        tmp[y,x]= 255
                    akazu = abs(hsv[0] - 90)
                    bkazu = bkazu + 1
                    TEKIOU = TEKIOU + akazu
                else:
                    
                    tmp[y,x]= 0

    return TEKIOU/bkazu

def main(args):
    
    "1. load original image"
    src = cv2.imread(args[1])
    src = cv2.imread(args[1])
    src_img_orig = src
    print("The Cyan Process: " ,args[1])
    alpha = 1.0
    size = src.shape
    
    "contrast-senkei"
    Max = [0,0,0]
    Min = [255,255,255]
    for y in range(0,size[0]):
        for x in range(0,size[1]):
            for c in range(0,size[2]):
                if Max[c] < src[y,x][c]:
                    Max[c] = src[y,x][c]
                if Min[c] > src[y,x][c]:
                    Min[c] = src[y,x][c]

    for y in range(0,size[0]):
        for x in range(0,size[1]):
            for c in range(0,size[2]):
                src[y,x][c] = (255 / (Max[c] - Min[c])) * src[y,x][c]
    
    "contrast-sekiwa"
    alpha = 1.0
    AVR = [0,0,0]
    CC = [0,0,0]
    beta = [0,0,0]
    new2_img = np.zeros([size[0], size[1]])
    for y in range(0,size[0]):
        for x in range(0,size[1]):
            for c in range(0,size[2]):
                AVR[c] = AVR[c] + src[y,x][c]
                CC[c] = CC[c] + 1
                
    for i in range(0,3):
        beta[i] = AVR[i]/CC[i];		    
        print(beta[i])
    
    for y in range(0,size[0]):
        for x in range(0,size[1]):
            for c in range(0,size[2]):
                src[y,x][c] = (alpha * (src[y,x][c]-beta[c])+beta[c])
                
    "senneika"
    k = 1.0
    sharpningKernel8 = np.array([[k, k, k],[k, 1 + (8*k*-1), k],[k, k, k]])
    dst = cv2.filter2D(src, -1,sharpningKernel8)

    "sironuki"
    Hue = 0.0
    white = np.zeros([size[0], size[1]])
    Hue = colorExtraction1(src, white,0,255,  0,120,  140,255)
    print(Hue)
    "irotyuusyutu"
    x_Cyan = 0.0
    y_Cyan = 0.0
    x_Cyan = Hue
    y_Cyan = (-1.043 * x_Cyan) + 186.44
    img_dst = np.zeros([size[0], size[1],size[2]])
    img_dst = colorInk(src, img_dst,90, 150 ,y_Cyan,255, 120, 255)
   
    "grayscalehennkann"
    img_dst = cv2.cvtColor(img_dst,cv2.COLOR_BGR2GRAY)
    "tekiounitika"
    img_dst = cv2.adaptiveThreshold(img_dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,101,0)
    "nitika"
    ret, img_dst = cv2.threshold(img_dst,127,255,cv2.THRESH_BINARY)
    "bithanntenn"
    cv2.imwrite(args[2], img_dst)
    return None

main(args)
