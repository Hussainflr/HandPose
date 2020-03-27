import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, sys







def blend(image, mask, alpha):

    beta = (1 - alpha)

    blendedImage = cv2.addWeighted(mask, alpha, image, beta, 0.0)



    return blendedImage





def main():
    imagedir = 'C:/Users/muzam/ori'
    maskdir = 'C:/Users/muzam/Updated_Crop_psp_res50'
    savedir = 'C:/Users/muzam/blendedResults'
    imagelist = os.listdir(imagedir)
    masklist = os.listdir(maskdir)



    for imgname in imagelist:
        imgpath = os.path.join(imagedir, imgname)

        maskname = imgname.replace(".jpeg", ".png")
        maskpath = os.path.join(maskdir, maskname)


        img  = cv2.imread(imgpath, 1)
        mask = cv2.imread(maskpath, 1)

        img = cv2.resize(img,(mask.shape[1], mask.shape[0]))
        print(mask.shape, img.shape)

        blendedimg = blend(img, mask, 0.5)


        cv2.imwrite(os.path.join(savedir, "blended_" + imgname), blendedimg)

    print("done..")




if __name__ == '__main__':
    main()



