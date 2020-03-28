import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
import numpy as np
from PIL import Image
from enum import Enum
class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)

class UnifiedVisualization:

    
    def __init__(self):

        self.fig, self.ax = plt.subplots(1)

        #self.ax = fig.add_subplot(111, projection='3d')
        #self.ax.set_xlabel('X axis')
        #self.ax.set_ylabel('Y axis')
        #self.ax.set_zlabel('Z axis')
        #self.ax = fig.add_subplot(2, 2, 2 + proj_idx)
    def plot_box(self, points):

        xs = points[:,0]
        ys = points[:,1]
        zs = points[:,2]
        self.ax.scatter3D(xs, ys, zs)    

        for i, (x, y, z) in enumerate(points):
            self.ax.text(x, y, z, str(i), color='red')
        
        # lines (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)
        # list of sides' polygons of figure
        verts = [[points[0],points[1],points[2],points[3]],
        [points[4],points[5],points[6],points[7]], 
        [points[0],points[1],points[5],points[4]], 
        [points[2],points[3],points[7],points[6]], 
        [points[1],points[2],points[6],points[5]],
        [points[4],points[7],points[3],points[0]]]

        # plot sides
        self.ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        


    def draw_box(self, imgpath, points, control_points):
        cam_intr = np.array([[1395.749023, 0, 935.732544],
                             [0, 1395.749268, 540.681030], [0, 0, 1]])



        img = cv2.imread(imgpath)



        box_3d = self.get_box_3d_coords(points, cam_intr)
        minpt, maxpt = self.get_rect_coords(box_3d)


        imgname = self.get_image_name(imgpath)



        path = imgpath.replace(imgname+'.jpeg', "")
        imgname = imgname.replace("color_", "")

        f = open(path + "bb.txt", "a")

        f.write(imgname + " " + str(minpt[0]) + " " + str(minpt[0]) + " " + str(maxpt[0]) + " " + str(maxpt[1])+ "\n")
        f.close()


        print(path)
        print(imgname)


        # img = self.plot_3d_box(img, box_3d)
        # img  = self.plot_rectangle(img, box_3d)
        # plt.imshow(img)
        #
        #
        #
        # plt.show()




    def get_image_name(self, imgpath):
        imgpath = imgpath.split('\\')

        imgpath = imgpath[len(imgpath)-1]
        imgpath = imgpath.split('.')

        imgname = imgpath[0]



        return imgname

    # takes in a 3d point and projects it into 2d

    def project_3d_pt(self, pt, cam_to_img, calib_file=None):
        # if calib_file is not None:
        #     cam_to_img = get_calibration_cam_to_image(calib_file)
        #     R0_rect = get_R0(calib_file)
        #     Tr_velo_to_cam = get_tr_to_velo(calib_file)

        cam_to_img = np.concatenate((cam_to_img, np.zeros([cam_to_img.shape[0],1])), axis=1)


        point = np.array(pt)
        point = np.append(point, 1)





        point = np.dot(cam_to_img, point)
        # point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

        point = point[:2] / point[2]
        point = point.astype(np.int16)


        return point

    def get_box_3d_coords(self, corners, cam_to_img):
        box_3d = []
        for corner in corners:
            point = self.project_3d_pt(corner, cam_to_img)
            box_3d.append(point)

        return box_3d



    def get_rect_coords(self, box_3d):
        minpt = tuple(np.min(box_3d, axis=0))

        maxpt = tuple(np.max(box_3d, axis=0))

        return minpt, maxpt






    def plot_3d_box(self, img, box_3d):

        # plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)

        #R = self.rotation_matrix(ry)


        #corners = self.create_corners(dimension, location=center, R=R)

        # to see the corners on image as red circles
        # plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False)



        #box_3d = self.project_3d_pt(corners, cam_to_img)

        # TODO put into loop
        cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0], box_3d[2][1]), cv_colors.GREEN.value, 5)
        cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0], box_3d[6][1]), cv_colors.GREEN.value, 5)
        cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0], box_3d[4][1]), cv_colors.GREEN.value, 5)

        #front left
        cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0], box_3d[6][1]), cv_colors.GREEN.value, 5)

        cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0], box_3d[3][1]), cv_colors.GREEN.value, 5)
        cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0], box_3d[5][1]), cv_colors.GREEN.value, 5)

        #front right
        cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0], box_3d[3][1]), cv_colors.GREEN.value, 5)
        cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0], box_3d[5][1]), cv_colors.GREEN.value, 5)

        for i in range(0, 7, 2):
            cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i + 1][0], box_3d[i + 1][1]), cv_colors.GREEN.value, 5)

        front_mark = [(box_3d[i][0], box_3d[i][1]) for i in range(4)]

        cv2.line(img, front_mark[0], front_mark[3], cv_colors.BLUE.value, 5)
        cv2.line(img, front_mark[1], front_mark[2], cv_colors.BLUE.value, 5)






        return img



#this function is purely from me

    def draw2DBox (self,img,  box_3d):

        box_3d = np.array(box_3d)
        scale = 1.2
        left_pt1 =  (int(box_3d[2][0]/scale), int(box_3d[2][1] * scale))
        pt1 = (int(box_3d[2][0]/scale), int(box_3d[2][1] / scale))

        pt1 = (int(box_3d[2][0]/ 1.1), int(box_3d[2][1] * 1.1))


        pt2 = (int(box_3d[6][0] / 1.1), int(box_3d[6][1] * 0.65 ))


        pt3 =  (int(box_3d[7][0] * 1.1 ), int(box_3d[7][1] * 0.65))

        pt4 =  (int(box_3d[3][0] * 1.1), int(box_3d[3][1] * 1.1 ))




        #left side
        #cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0], box_3d[6][1]), cv_colors.GREEN.value, 5)
        cv2.line(img, pt1, pt2, cv_colors.ORANGE.value, 5)

        #right side
        #cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0], box_3d[3][1]), cv_colors.GREEN.value, 5)
        cv2.line(img, pt3, pt4, cv_colors.ORANGE.value, 5)
        #bottom
        #cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[3][0], box_3d[3][1]), cv_colors.ORANGE.value, 5)
        cv2.line(img, pt1, pt4, cv_colors.ORANGE.value, 5)
        #top
        #cv2.line(img, (box_3d[6][0], box_3d[6][1]), (box_3d[7][0], box_3d[7][1]), cv_colors.ORANGE.value, 5)
        cv2.line(img, pt2, pt3, cv_colors.ORANGE.value, 5)



        #self.plot_circles(img, pt1, pt2, pt3, pt4, minpt, maxpt)

        return img




    def plot_rectangle(self, img, box_3d):

        minpt, maxpt = self.get_rect_coords(box_3d)

        cv2.rectangle(img, minpt, maxpt, cv_colors.YELLOW.value, thickness=10)

        return img




    def plot_circles(self, img, pt1, pt2, pt3, pt4):

        pt11 = (609, 150)
        pt22 = (1161, 954)
        cv2.circle(img, pt1, 10, cv_colors.PURPLE.value, thickness=10)
        cv2.circle(img, pt2, 10, cv_colors.PURPLE.value, thickness=10)

        cv2.circle(img, pt3, 10, cv_colors.PURPLE.value, thickness=10)
        cv2.circle(img, pt4, 10, cv_colors.PURPLE.value, thickness=10)








        return img




    # using this math: https://en.wikipedia.org/wiki/Rotation_matrix
    def rotation_matrix(self, yaw, pitch=0, roll=0):
        tx = roll
        ty = yaw
        tz = pitch

        Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
        Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
        Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])

        return Ry.reshape([3, 3])
        # return np.dot(np.dot(Rz,Ry), Rx)








    # option to rotate and shift (for label info)
    def create_corners(self, dimension, location=None, R=None):
        dx = dimension[2] / 2
        dy = dimension[0] / 2
        dz = dimension[1] / 2

        x_corners = []
        y_corners = []
        z_corners = []

        for i in [1, -1]:
            for j in [1, -1]:
                for k in [1, -1]:
                    x_corners.append(dx * i)
                    y_corners.append(dy * j)
                    z_corners.append(dz * k)

        corners = [x_corners, y_corners, z_corners]

        # rotate if R is passed in
        if R is not None:
            corners = np.dot(R, corners)

        # shift if location is passed in
        if location is not None:
            for i, loc in enumerate(location):
                corners[i, :] = corners[i, :] + loc

        final_corners = []
        for i in range(8):
            final_corners.append([corners[0][i], corners[1][i], corners[2][i]])

        return final_corners










































    def plot_hand(self, points):

        thumb = points[1:5]
        index = points[5:9]
        middle = points[9:13]
        ring = points[13:17]
        pinky = points[17:21]

        thumb_xs = thumb[:,0]
        thumb_ys = thumb[:,1]
        thumb_zs = thumb[:,2]
        self.ax.plot(thumb_xs, thumb_ys, thumb_zs)

        index_xs = index[:,0]
        index_ys = index[:,1]
        index_zs = index[:,2]
        self.ax.plot(index_xs, index_ys, index_zs)

        middle_xs = middle[:,0]
        middle_ys = middle[:,1]
        middle_zs = middle[:,2]
        self.ax.plot(middle_xs, middle_ys, middle_zs)

        ring_xs = ring[:,0]
        ring_ys = ring[:,1]
        ring_zs = ring[:,2]
        self.ax.plot(ring_xs, ring_ys, ring_zs)

        pinky_xs = pinky[:,0]
        pinky_ys = pinky[:,1]
        pinky_zs = pinky[:,2]
        self.ax.plot(pinky_xs, pinky_ys, pinky_zs)

        wrist_x = points[0,0]
        wrist_y = points[0,1]
        wrist_z = points[0,2]

        thumb_x = points[1,0]
        thumb_y = points[1,1]
        thumb_z = points[1,2]

        self.ax.plot([wrist_x, thumb_x], [wrist_y, thumb_y], [wrist_z, thumb_z])

        index_x = points[5,0]
        index_y = points[5,1]
        index_z = points[5,2]

        self.ax.plot([wrist_x, index_x], [wrist_y, index_y], [wrist_z, index_z])

        middle_x = points[9,0]
        middle_y = points[9,1]
        middle_z = points[9,2]

        self.ax.plot([wrist_x, middle_x], [wrist_y, middle_y], [wrist_z, middle_z])

        ring_x = points[13,0]
        ring_y = points[13,1]
        ring_z = points[13,2]

        self.ax.plot([wrist_x, ring_x], [wrist_y, ring_y], [wrist_z, ring_z])

        pinky_x = points[17,0]
        pinky_y = points[17,1]
        pinky_z = points[17,2]

        self.ax.plot([wrist_x, pinky_x], [wrist_y, pinky_y], [wrist_z, pinky_z])

    


    # def hand_plot(self, img, skel_camcoords):
    #     self.ax = fig.add_subplot(221)
    #     self.ax.imshow(img)
    #     for proj_idx, (proj_1, proj_2) in enumerate([[0, 1], [1, 2], [0, 2]]):
    #         self.ax = fig.add_subplot(2, 2, 2 + proj_idx)
    #         if proj_idx == 0:
    #             # Invert y axes to align with image in camera projection
    #             self.ax.invert_yaxis()
    #         self.ax.set_aspect('equal')
    #         if args.obj is not None:
    #             self.ax.scatter(
    #                 verts_camcoords[:, proj_1], verts_camcoords[:, proj_2], s=1)
    #         self.visualize_joints_2d(
    #             self.ax,
    #             np.stack(
    #                 [skel_camcoords[:, proj_1], skel_camcoords[:, proj_2]],
    #                 axis=1),
    #             joint_idxs=False)



    def plot(self):

        plt.show()
    def saveFig(self, count):

        #plt.savefig('/SSD2/hussain/Results/H+O/result'+str(count)+'.png')
        plt.savefig('../results/result'+str(count)+'.png')
    