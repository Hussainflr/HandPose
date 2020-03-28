import os, sys
import pickle

import torch
import trimesh
import numpy as np
from PIL import Image
import copy
import cv2

from torchvision import transforms
from torchvision import utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import torchvision
import json
import tqdm


class UnifiedPoseDataset(Dataset):

    def __init__(self, mode='train', root='../data/First_Person_Action_Benchmark', loadit=False, name=None):

        self.loadit = loadit
        self.temp = 0

        if name is None:
            self.name = mode
        else:
            self.name = name

        self.root = root

        if mode == 'clean':
            self.subjects = [1]

        elif mode == 'test2':
            self.subjects = [2, 5, 4]
        else:
            raise Exception("Incorrect vallue for for 'mode': {}".format(mode))

        subject = "Subject_1"
        subject = os.path.join(root, 'Object_6D_pose_annotation_v1', subject)
        self.actions = os.listdir(subject)
        print(self.actions)
        self.object_names = ['juice', 'liquid_soap', 'milk', 'salt']

        action_to_object = {
            'open_milk': 'milk',
            'close_milk': 'milk',
            'pour_milk': 'milk',
            'open_juice_bottle': 'juice',
            'close_juice_bottle': 'juice',
            'pour_juice_bottle': 'juice',
            'open_liquid_soap': 'liquid_soap',
            'close_liquid_soap': 'liquid_soap',
            'pour_liquid_soap': 'liquid_soap',
            'put_salt': 'salt'
        }

        categories_ids = {
            'grasp': 1,
            'wrap-grasp': 2,
            'contain': 3,
            'openable': 4,
            'cleanable': 5,
            'squeezable': 6,
            'drinkable': 7,
            'dippable': 8,
            'readable': 9,
            'scoop': 10

        }
        root_dir = "E:\Research\Important\Dataset\FPHA-Afford\\New_FPAB-Afford"

        if not loadit:
            check = []
            self.temp = 0
            self.samples = {}
            idx = 0
            image_count = 0
            # folder count
            fcount = 0

            # count coco annotation files
            anno_count = 0

            # count annotated images
            anno_image_count = 0
            train_dic = {}
            test_dic = {}

            train_count = 0
            test_count = 0
            action_count = 0
            actionname = ''

            testflag = False

            for subject in self.subjects:

                subject = "Subject_" + str(subject)
                flist = os.listdir(os.path.join(root, 'Video_files', subject))

                for fname in flist:
                    video_sequences = os.listdir(os.path.join(root, 'Video_files', subject, fname))

                    for vs in video_sequences:

                        frames = os.listdir(os.path.join(root, 'Video_files', subject, fname, vs, 'color'))

                        # if you want iterate through all iamges then remove the condition "if frames.__contains__("via_export_coco.json"):"
                        if frames.__contains__("via_export_coco.json"):
                            if action_count == 0 or actionname != fname:
                                actionname = fname
                                print(actionname)
                                action_count += 1

                            if int(vs) == 2:
                                testflag = True
                                image_count = test_count

                            else:
                                testflag = False
                                image_count = train_count

                            anno_path = os.path.join(root, 'Video_files', subject, fname, vs, 'color',
                                                     'via_export_coco.json')
                            with open(anno_path, 'r') as f:
                                data = dict(json.load(f))
                                f.close()

                            # let's build single coco annotation file

                            anno_count += 1
                            idx = 0
                            annlist = []
                            imagelist = []
                            catlist = []

                            # get categories folderwise
                            catdic = {}
                            for cat in data['categories']:
                                catdic.__setitem__(cat['id'], cat['name'])
                                cat['id'] = categories_ids.get(cat['name'])
                                catlist.append(cat)

                            for frame in frames:

                                # file could be an image or an annotation file. Both are specified with extension
                                image_path = os.path.join(root, 'Video_files', subject, fname, vs, 'color', frame)
                                depth_path = os.path.join(root, 'Video_files', subject, fname, vs, 'depth')

                                if frame.__contains__('.jpeg'):

                                    # counting total images

                                    # this counter is for annotated images
                                    # c += 1

                                    # img = cv2.imread(image_path, 1)

                                    # subject number
                                    sname = str(subject).replace('Subject_', '')
                                    # frame name

                                    fn = str(frame).replace('.jpeg', '').replace('color_', '')

                                    idx = int(fn)

                                    # # updated frame name
                                    # new_frame_name = sname + str(fcount) + str(vs) + fn
                                    #
                                    # imgdir = os.path.join(train_dir, new_frame_name)
                                    # # cv2.imwrite(imgdir, img)


                                    pt1 = []
                                    pt2 = []


                                    for ann in data['annotations']:

                                        # changing image_id into to Int because VIA annotator gives it in string but in coco format it must be in Int
                                        image_id = int(ann['image_id'])

                                        if idx == image_id:
                                            # updated frame name
                                            new_frame_name = sname + str(fcount) + str(vs) + fn

                                            #ann['image_id'] = int(new_frame_name)

                                            seg = ann['segmentation']

                                            #ann['segmentation'] = [seg]

                                            poly = np.asarray(seg).reshape((int(len(seg) / 2), 2))


                                            pt1.append(np.min(poly, axis=0).astype(dtype=int))
                                            pt2.append(np.max(poly, axis=0).astype(dtype=int))



                                            #
                                            #
                                            #
                                            # ann['id'] = image_count
                                            #
                                            # c = ann['category_id']
                                            #
                                            # # get cat name
                                            # catname = catdic.get(c)
                                            #
                                            # # get cat id from defined dictionary "categories_ids"
                                            # try:
                                            #     cat_id = categories_ids[catname]
                                            #
                                            # except NameError:
                                            #     print(sys.exc_info())
                                            #     raise NameError("Category ID is not correct", catname)
                                            #
                                            # # update cat id in ann['category_id']
                                            # ann['category_id'] = cat_id
                                            #
                                            # annlist.append(ann)
                                            #
                                            # if check.__contains__(int(new_frame_name)):
                                            #     print(new_frame_name)
                                            #     raise ValueError("Repitation.. occurs")
                                            # else:
                                            #
                                            #     check.append(new_frame_name)
                                            # depth_path = os.path.join(root, 'Video_files', subject, fname, vs, 'depth')
                                            # depth_name = 'depth_' + fn + '.png'
                                            # depth_path = os.path.join(depth_path, depth_name)

                                            # frame counters
                                            # if testflag:
                                            #     # read frame
                                            #     try:
                                            #         img = cv2.imread(image_path, 1)
                                            #
                                            #         depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                                            #
                                            #     except ValueError:
                                            #         print(depth_path)
                                            #         print(image_path)
                                            #         raise ValueError(
                                            #             "something is wrong in image reading folder check above folder")
                                            #
                                            #     imgdir = os.path.join(root_dir,'test',str(new_frame_name) + '.jpeg')
                                            #     depthdir = os.path.join(root_dir, 'depth_test',str(new_frame_name) + '.png')
                                            #     # save frame in new separat dir
                                            #     try:
                                            #         cv2.imwrite(imgdir, img)
                                            #         cv2.imwrite(depthdir, depth)
                                            #     except ValueError:
                                            #         print(type(img))
                                            #         print(imgdir)
                                            #         ValueError(
                                            #             "something is wrong in image save folder check above folder")
                                            #
                                            #     test_count += 1
                                            #
                                            # else:
                                            #     # read frame
                                            #     try:
                                            #         img = cv2.imread(image_path, 1)
                                            #
                                            #         depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                                            #
                                            #
                                            #
                                            #
                                            #
                                            #
                                            #     except ValueError:
                                            #         print(image_path)
                                            #         raise ValueError(
                                            #             "something is wrong in image reading folder check above folder")
                                            #
                                            #     imgdir = os.path.join(root_dir,'train',str(new_frame_name) + '.jpeg')
                                            #     depthdir = os.path.join(root_dir, 'depth_train',
                                            #                             str(new_frame_name) + '.png')
                                            #
                                            #     # save frame in new separat dir
                                            #     try:
                                            #         cv2.imwrite(imgdir, img)
                                            #         cv2.imwrite(depthdir, depth)
                                            #     except ValueError:
                                            #         print(type(img))
                                            #         print(imgdir)
                                            #         ValueError(
                                            #             "something is wrong in image save folder check above folder")
                                            #
                                            #     train_count += 1

                                            # mask counter
                                            # image_count += 1

                                    if len(pt1)> 0 and len(pt2) > 0:
                                        imgdir = os.path.join(root_dir, 'crop', str(new_frame_name) + '.jpeg')
                                        temimg = cv2.imread(image_path, 1)
                                        h, w, _ = temimg.shape


                                        pts1 = np.min(pt1, axis=0)
                                        pts2 = np.max(pt2, axis=0)
                                        center = (int((pts1[0] + pts2[0]) / 2), int((pts1[1] + pts2[1]) / 2))
                                        xscale = 350
                                        yscale = 300

                                        xr = int(xscale * 0.5)
                                        yr = int(yscale * 0.5)



                                        #cv2.circle(temimg, center, 20, (0, 0, 255), -1)

                                        dpts1 = [pts1[0] - xr, pts1[1] - yr]


                                        dpts2 = [pts2[0] + xscale, pts2[1] + yscale]


                                        if dpts1[0] < 0:
                                            dpts2[0] += int(abs(dpts1[0])/2) if dpts2[0] < temimg.shape[1] else temimg.shape[1]
                                            dpts2[1] += int(abs(dpts1[0]) / 2) if dpts2[1] < temimg.shape[0] else temimg.shape[0]
                                            if dpts2[0] > temimg.shape[1]:
                                                dpts2[0] = temimg.shape[1]
                                            if dpts2[0] > temimg.shape[0]:
                                                dpts2[0] = temimg.shape[0]

                                            dpts1[0] = 0

                                        if dpts1[1] < 0:
                                            dpts2[1] += int(abs(dpts1[1])/2) if dpts2[1] < temimg.shape[0] else temimg.shape[0]
                                            dpts2[0] += int(abs(dpts1[1]) / 2) if dpts2[0] < temimg.shape[1] else temimg.shape[1]
                                            if dpts2[1] > temimg.shape[0]:
                                                dpts2[1] = temimg.shape[0]

                                            if dpts2[0] > temimg.shape[1]:
                                                dpts2[0] = temimg.shape[1]

                                            dpts1[1] = 0



                                        if dpts2[0] > temimg.shape[1]:
                                            dpts1[0] -= int((int(dpts2[0] - temimg.shape[1])/2)/2) if dpts1[0] > 0 else 0
                                            dpts1[1] -= int(((int(dpts2[0] - temimg.shape[1]) / 2)) / 2) if dpts1[1] > 0 else 0
                                            if dpts1[0] < 0:
                                                dpts1[0] = 0
                                            if dpts1[1] < 0:
                                                dpts1[1] = 0

                                            dpts2[0] = temimg.shape[1]
                                            print(dpts2)
                                            print(3, dpts1[0])
                                        if dpts2[1] > temimg.shape[0]:
                                            print(4)
                                            dpts1[1] -= int(((dpts2[1] - temimg.shape[0])/2)/2) if dpts1[1] > 0 else 0
                                            dpts1[0] -= int(((dpts2[1] - temimg.shape[0]) / 2) / 2) if dpts1[0] > 0 else 0
                                            if dpts1[1] < 0:
                                                dpts1[1] = 0
                                            if dpts1[0] < 0:
                                                dpts1[0] = 0
                                            dpts2[1] = temimg.shape[0]
                                            #dpts1[1] -= int(/2) if dpts1[1] < 0 else 0


                                        # print("All:", dpts1, dpts2
                                        #       )

                                        # cv2.rectangle(temimg, tuple(pts1), tuple(pts2), (0, 255, 0), thickness=10)
                                        # cv2.rectangle(temimg, tuple(dpts1), tuple(dpts2), (255, 0, 0), thickness=10)

                                        mask = np.zeros((h, w))



                                        #crop image and depth
                                        cropimg = temimg[dpts1[1]: dpts2[1],dpts1[0]: dpts2[0] ]

                                        depth_name = 'depth_' + fn + '.png'
                                        depth_path = os.path.join(depth_path, depth_name)
                                        depthmap = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

                                        cv2.rectangle(depthmap, tuple(dpts1), tuple(dpts2), (255, 0, 0), thickness=10)
                                        cropdepth = depthmap #[dpts1[1]:dpts2[1], dpts1[0]:dpts2[0]]

                                        ch, cw, _ = cropimg.shape




                                        xratio = w/cw
                                        yratio = h/ch








                                    for ann in data['annotations']:

                                        # changing image_id into to Int because VIA annotator gives it in string but in coco format it must be in Int
                                        image_id = int(ann['image_id'])

                                        if idx == image_id:
                                            # updated frame name
                                            new_frame_name = sname + str(fcount) + str(vs) + fn

                                            ann['image_id'] = int(new_frame_name)

                                            seg = ann['segmentation']
                                            print(seg)

                                            for i in range(len(seg)):
                                                if i % 2 == 0:

                                                    seg[i] /=yratio
                                                else:
                                                    seg[i] /=xratio

                                            print(seg)



                                            ann['segmentation'] = [seg]

                                            # poly = np.asarray(seg).reshape((int(len(seg) / 2), 2))

                                            # print("Poly", poly)
                                            # print(poly[0])
                                            # pt1.append(np.min(poly, axis=0).astype(dtype=int))
                                            # pt2.append(np.max(poly, axis=0).astype(dtype=int))






                                            ann['id'] = image_count

                                            c = ann['category_id']

                                            # get cat name
                                            catname = catdic.get(c)

                                            # get cat id from defined dictionary "categories_ids"
                                            try:
                                                cat_id = categories_ids[catname]

                                            except NameError:
                                                print(sys.exc_info())
                                                raise NameError("Category ID is not correct", catname)

                                            # update cat id in ann['category_id']
                                            ann['category_id'] = cat_id

                                            annlist.append(ann)

                                            if check.__contains__(int(new_frame_name)):
                                                print(new_frame_name)
                                                raise ValueError("Repitation.. occurs")
                                            else:

                                                check.append(new_frame_name)



                                            # # frame counters
                                            # if testflag:
                                            #     # read frame
                                            #     try:
                                            #         img = cropimg
                                            #
                                            #
                                            #         depth = cropdepth
                                            #
                                            #     except ValueError:
                                            #         print(depth_path)
                                            #         print(image_path)
                                            #         raise ValueError(
                                            #             "something is wrong in image reading folder check above folder")
                                            #
                                            #     imgdir = os.path.join(root_dir,'test',str(new_frame_name) + '.jpeg')
                                            #     depthdir = os.path.join(root_dir, 'depth_test',str(new_frame_name) + '.png')
                                            #     # save frame in new separat dir
                                            #     try:
                                            #         cv2.imwrite(imgdir, img)
                                            #         cv2.imwrite(depthdir, depth)
                                            #     except ValueError:
                                            #         print(type(img))
                                            #         print(imgdir)
                                            #         ValueError(
                                            #             "something is wrong in image save folder check above folder")
                                            #
                                            #     test_count += 1
                                            #
                                            # else:
                                            #     # read frame
                                            #     try:
                                            #         img = cropimg
                                            #
                                            #         depth = cropdepth
                                            #
                                            #
                                            #
                                            #
                                            #
                                            #
                                            #     except ValueError:
                                            #         print(image_path)
                                            #         raise ValueError(
                                            #             "something is wrong in image reading folder check above folder")
                                            #
                                            #     imgdir = os.path.join(root_dir,'train',str(new_frame_name) + '.jpeg')
                                            #     depthdir = os.path.join(root_dir, 'depth_train',
                                            #                             str(new_frame_name) + '.png')
                                            #
                                            #     # save frame in new separat dir
                                            #     try:
                                            #         cv2.imwrite(imgdir, img)
                                            #         cv2.imwrite(depthdir, depth)
                                            #     except ValueError:
                                            #         print(type(img))
                                            #         print(imgdir)
                                            #         ValueError(
                                            #             "something is wrong in image save folder check above folder")
                                            #
                                            #     train_count += 1

                                            # mask counter
                                            image_count += 1




                                    for element in data['images']:

                                        # changing image_id into to Int because VIA annotator gives it in string but in coco format it must be in Int
                                        id = int(element['id'])

                                        if idx == id:
                                            # updated frame name

                                            element['id'] = int(new_frame_name)
                                            element['file_name'] = str(new_frame_name) + '.jpeg'
                                            imagelist.append(element)

                            if train_dic.__len__() == 0 or test_dic.__len__() == 0:

                                if testflag:
                                    test_dic = data
                                    test_dic['annotations'] = annlist
                                    test_dic['images'] = imagelist
                                    test_dic['categories'] = catlist
                                    with open(os.path.join(root_dir, 'test_tem.json'), 'w+') as f:
                                        print("%%%%%%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%%%%%%%%%%")
                                        json.dump(test_dic, f)
                                        exit()
                                else:
                                    train_dic = data
                                    train_dic['annotations'] = annlist
                                    train_dic['images'] = imagelist
                                    train_dic['categories'] = catlist
                                    with open(os.path.join(root_dir, 'train_tem.json'), 'w+') as f:
                                        print("***************Training********************")
                                        json.dump(train_dic, f)





                            else:

                                if testflag:
                                    annotations = test_dic['annotations']
                                    images = test_dic['images']

                                    for ann in annlist:
                                        annotations.append(ann)

                                    test_dic['annotations'] = annotations

                                    for element in imagelist:
                                        images.append(element)

                                    test_dic['images'] = images

                                    # update categories in annotation file

                                    categories = test_dic['categories']
                                    for cat in catlist:
                                        if not categories.__contains__(cat):
                                            categories.append(cat)
                                    print("%%%%%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%%%%%%%%%%%")



                                else:
                                    annotations = train_dic['annotations']
                                    images = train_dic['images']

                                    for ann in annlist:
                                        annotations.append(ann)

                                    train_dic['annotations'] = annotations

                                    for element in imagelist:
                                        images.append(element)

                                    train_dic['images'] = images

                                    # update categories in annotation file

                                    categories = train_dic['categories']
                                    for cat in catlist:
                                        if not categories.__contains__(cat):
                                            categories.append(cat)
                                    print("***************Training********************")

                    # this count is useful in image naming and counting json files
                    fcount += 1

                    # anno_count += 1
                    # anno_image_count += c

                    # elif frame == 'via_export_coco.json':
                    #     with open(file, 'r') as f:
                    #         data = dict(json.load(f))
                    #     # let's build single coco annotation file
                    #     if train_dic.__len__() == 0:
                    #         train_dic = data
                    #         print(train_dic.keys())
                    #
                    #     # anno_count += 1
                    #     anno_image_count += c

                # print("Coco Annotation files: {} images count: {}".format( anno_count, anno_image_count))

            with open(os.path.join(root_dir, 'annotations', 'FPAB-Afford_Coco_Training.json'), 'w+') as f:
                print("****************Training*******************")
                json.dump(train_dic, f)
                f.close()
            with open(os.path.join(root_dir, 'annotations', 'FPAB-Afford_Coco_Testing.json'), 'w+') as f:
                print("%%%%%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%%%%%%%%%%%")
                json.dump(test_dic, f)
                f.close()
            print(
                "Image count: {} folder count: {} Annotation files count: {} Training Images: {} Testing Images: {} Action_count:{} ".format(
                    image_count, fcount, anno_count, train_count, test_count, action_count))

            print("Done...", len(check))
