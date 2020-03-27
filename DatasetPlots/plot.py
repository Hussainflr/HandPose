from matplotlib import pyplot as plt
import numpy as np
import cv2
import os





def resizer(path):

    dirlist = os.listdir(path)

    for f in dirlist:
        imglist = os.listdir(os.path.join(path, f))
        dstpath = os.path.join(path, f, 'resized')
        if not os.path.exists(dstpath):
            dstpath = os.mkdir(dstpath)
        for item in imglist:
            if item.__contains__('.jpeg'):
                imgpath = os.path.join(path, f, item)
                print(item)
                img = cv2.imread(imgpath, 1)
                img = cv2.resize(img, (1000, 600), interpolation=cv2.INTER_AREA)
                # item = item.split('.')[0]
                # print(item)
                # item = item+'.png'

                cv2.imwrite(os.path.join(dstpath, item), img)
        print(f,"done ...")


def show_barplot(cats, data, xlabel, ylabel, title):
    plt.style.use('fivethirtyeight')
    width = 0.25
    plt.barh(cats, data)

    plt.grid(None)
    # plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plt.tight_layout()
    #plt.title( title)

    plt.show()
    plt.savefig('masks_per_image.png')


# def data_mask_cat():
#     categories = ['grasp', 'wrap-grasp', 'contain', 'openable', 'cleanable', 'drinkable', 'readable', 'scoop']
#     values = [1536, 1491, 451, 1268, 486, 867, 267, 534]
#     print(sum(values))
#     xlabel = "Number of Annotation Masks"
#     ylabel = "Affordance Categories"

#     title = "Masks per frames"



#     return categories, values, xlabel, ylabel, title



def data_image_cat():
    #masks
    masks_categories = ['grasp', 'wrap-grasp', 'contain', 'openable', 'cleanable', 'drinkable', 'readable', 'scoop', 'dippable']
    masks_values = [1536, 1321, 351, 1268, 486, 528, 192, 534, 339]
    xlabel = "Number of Annotation Masks"
    ylabel = "Affordance Categories"
    title = "Masks per frames"





    return masks_categories, masks_values, xlabel, ylabel, title


def data_image_obj():
    obj_categories = ['mug', 'card', 'letter', 'liquid', 'milk', 'peanut', 'soda', 'wallet', 'juice', 'wine', 'sugar',
                      'tea bag', 'spoon', 'sponge']

    obj_values = [302, 252, 303, 186, 359, 292, 174, 163, 417, 526, 93, 339, 446, 440]
    obj_xlabel = "Number of frames"
    obj_ylabel = "Object Categories"

    obj_title = "frames per object"

    return obj_categories, obj_values, obj_xlabel, obj_ylabel, obj_title










def main():
    
    cats, values, xlabel,ylabel, title = data_image_cat()
    print(cats)
    
    show_barplot(cats,values, xlabel, ylabel, title)

    # resizer('C:\\Users\muzam\masks')




if __name__=="__main__":
    main()