from tqdm import tqdm
import torch

import numpy as np
import sys
import os


#from cfg import parameters
from upe.net import UnifiedNetwork
from upe.dataset_o import UnifiedPoseDataset
from upe.visualize import UnifiedVisualization



if __name__ == '__main__':
    training_dataset = UnifiedPoseDataset(mode = 'test2', loadit=True, name='test2')
    training_dataset[0]
    print(training_dataset.samples[0])
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=False, num_workers=1)

    # model = UnifiedNetwork()
    # model.load_state_dict(torch.load('../models/unified_net.pth'))
    # model.eval()
    # model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr)

    # validation

    with torch.no_grad():

        hand_cell_counter = 0.
        object_cell_counter = 0.
        object_counter = 0.
        action_counter = 0.


        avgTotalloss = 0
        avgPoseloss = 0
        avgConfloss = 0
        avgActloss = 0
        avgObjloss = 0
        count = 1
        for batch, data in enumerate(tqdm(training_dataloader)):
            image = data[0]
            image_path = data[1]
            control_points = data[2]
            true = [x.cuda() for x in data[3:]]

            # print("####### iteration Number: ",count," #######")
            # count = count+1
            # pred = model(image.cuda())
            # #loss = loss + model.total_loss(pred, true)
            # #computing the Total loss
            # #tl= Total loss, pl= Pose loss, cl= Conf loss, al = Action loss, ol = Object loss
            # tl, pl, cl, al, ol = model.total_loss(pred, true)
            # avgTotalloss = avgTotalloss + tl
            # print("Loss: ", tl.data.cpu().numpy())
            #
            # pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = [
            #     p.data.cpu().numpy() for p in pred]
            true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = [
                t.data.cpu().numpy() for t in true]
            # #Computing the pose loss
            # avgPoseloss = avgPoseloss + pl
            # print("Pose Loss: ", avgPoseloss.data.cpu().numpy())
            # true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
            true_object_cell = np.unravel_index(object_mask.argmax(), object_mask.shape)
            #
            # #computing the Conf loss
            # avgConfloss= avgConfloss + cl
            # print("Conf loss: ", cl.data.cpu().numpy())
            #
            # pred_hand_cell = np.unravel_index(pred_hand_conf.argmax(), pred_hand_conf.shape)
            # pred_object_cell = np.unravel_index(pred_object_conf.argmax(), pred_object_conf.shape)
            #
            # #computing the Action loss
            # avgActloss = avgActloss + al
            # print("Action Loss: ", al.data.cpu().numpy())
            #
            # #computing the Object loss
            # avgObjloss = avgObjloss + ol
            # print("Object Loss: ", ol.data.cpu().numpy())
            #
            # hand_cell_counter += int(true_hand_cell == pred_hand_cell)
            # object_cell_counter += int(true_object_cell == pred_object_cell)
            #
            # z, v, u = true_hand_cell[1:]
            # # print('True Hand Pose: ', true_hand_pose.shape)
            # # print('Pred_hand Pose: ',  pred_hand_pose.shape)
            # # print('This is z :', z)
            # # print('This is v :', v)
            # # print('This is u :', u)
            # # print('Check below')
            # dels = pred_hand_pose[0, :, z, v, u].reshape(21, 3)
            #
            #
            # del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
            # print(del_u.shape, del_v.shape, del_z.shape)
            # hand_points = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))
            #
            # # Edit START
            # dels2 = true_hand_pose[0, :, z, v, u].reshape(21, 3)
            #
            # del_u2, del_v2, del_z2 = dels2[:, 0], dels2[:, 1], dels2[:, 2]
            # # print('Below are Hand Pose UVZ points:')
            # # print(del_u2.shape, del_v2.shape, del_z2.shape)
            # hand_points2 = training_dataset.target_to_control(del_u2, del_v2, del_z2, (u, v, z))
            # #edit end
            #
            z, v, u = true_object_cell[1:]
            # dels = pred_object_pose[0, :, z, v, u].reshape(21, 3)
            #
            #
            # del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
            # object_points = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))

            #Edited start
            dels2 = true_object_pose[0, :, z, v, u].reshape(21, 3)
            del_u2, del_v2, del_z2 = dels2[:, 0], dels2[:, 1], dels2[:, 2]
            # print('Below are the Object Pose UVZ points:')
            # print(del_u2.shape, del_v2.shape, del_z2.shape)
            object_points1 = training_dataset.target_to_control(del_u2, del_v2, del_z2, (u, v, z))
            #edit end
            #print(training_dataset.samples[batch])
            # print("below is the shape of Hand and Object :")
            # print('Predicted',hand_points.shape)
            # print('Original',hand_points2.shape)
            # print('predicted',object_points.shape)
            # print('Original',object_points1.shape)
            viz = UnifiedVisualization()
            # viz.plot_hand(hand_points2)
            # viz.plot_box(object_points1[1:9, :])
            # viz.saveFig(count)



            viz.draw_box(image_path[0], object_points1[1:9, :], control_points)



            #viz.plot()
