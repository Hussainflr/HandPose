import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import sys
import numpy as np
from PIL import Image
import torch.utils.data.sampler as Sampler
sys.path.append("..")
#I have changed Yml file you can refer to authors in case somthing is going wrong
from upe.cfg import parameters

from matplotlib import pyplot as plt





class UnifiedNetwork(nn.Module):

    def __init__(self, n_class=21, pretrained=True):

        super(UnifiedNetwork, self).__init__()

        self.num_hand_control_points = parameters.num_hand_control_points
        self.num_object_control_points = parameters.num_object_control_points
        self.num_actions = parameters.num_actions
        self.num_objects = parameters.num_objects
        self.depth_discretization = parameters.depth_discretization
        self.magicnum = 9 #this magic number is added from me not from auther this could be offset

        #YOlO9000 pretrained dict or Darknet-19
        yoloweights= '../weights/darknet19.pth'



        model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-2])

        for param in self.features.parameters():
            param.requires_grad = False

        self.hand_vector_size = 3 * self.num_hand_control_points + 1 + self.num_actions
        self.object_vector_size = 3 * self.num_object_control_points + 1 + self.num_objects
        self.target_channel_size = self.depth_discretization * (self.hand_vector_size + self.object_vector_size)
        print("These are target channel", self.target_channel_size)
        print("hand Vector size and Object Vector size:", self.hand_vector_size,  self.object_vector_size)
        # prediction layers'
        self.conv = nn.Conv2d(512, self.target_channel_size, (3, 3), padding=1)

        # ***From here I have added new code

        # conv1
        self.conv1_1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.LeakyReLU(0.1, inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)


        self.conv1_2 = nn.Conv2d(32, 64, 3, stride=1,padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.LeakyReLU(0.1, inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)


        # conv2

        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2_2 = nn.Conv2d(128, 64, 1, stride=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.LeakyReLU(0.1, inplace=True)

        # conv3
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.relu3_1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv3_2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv3_3 = nn.Conv2d(256, 128, 1, stride=1)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.relu3_3 = nn.LeakyReLU(0.1, inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv4
        self.conv4_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.relu4_1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv4_2 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv4_3 = nn.Conv2d(512, 256, 1, stride=1)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.relu4_3 = nn.LeakyReLU(0.1, inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv5
        self.conv5_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv5_2 = nn.Conv2d(512, 256, 1, stride=1)
        self.bn5_2 = nn.BatchNorm2d(256)
        self.relu5_2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv_rout16 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn_rout16 = nn.BatchNorm2d(512)
        self.relu_rout16 = nn.LeakyReLU(0.1, inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Conv6

        self.conv6_1 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.bn6_1 = nn.BatchNorm2d(1024)
        self.relu6_1 = nn.LeakyReLU(0.1, inplace=True)


        self.conv6_2 = nn.Conv2d(1024, 512, 1, stride=1)
        self.bn6_2 = nn.BatchNorm2d(512)
        self.relu6_2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv6_3 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.bn6_3 = nn.BatchNorm2d(1024)
        self.relu6_3 = nn.LeakyReLU(0.1, inplace=True)

        self.conv7_1 = nn.Conv2d(1024, 512, 1, stride=1)
        self.bn7_1 = nn.BatchNorm2d(512)
        self.relu7_1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv7_2 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.bn7_2 = nn.BatchNorm2d(1024)
        self.relu7_2 = nn.LeakyReLU(0.1, inplace=True)


        self.conv7_3 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.bn7_3 = nn.BatchNorm2d(1024)
        self.relu7_3 = nn.ReLU(inplace=True)

        self.conv8_1 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.bn8_1 = nn.BatchNorm2d(1024)
        self.relu8_1 = nn.LeakyReLU(0.1, inplace=True)
        # self.conv8_2 = nn.Conv2d(1024, 1024, 3, stride=1)
        # self.conv8_2 = nn.ReLU(inplace=True)

        self.conv_rout26 = nn.Conv2d(512, 64, 1, stride=1)
        self.bn_rout26 =nn.BatchNorm2d(64)
        self.relu_rout26 = nn.LeakyReLU(0.1, inplace=True)


        # Resuming comments: I was implementing last two layer of network:
        self.convf_1 = nn.Conv2d(1280, 1024, 3, stride=1, padding=1)
        self.bnf_1 = nn.BatchNorm2d(1024)
        self.reluf_1 = nn.LeakyReLU(0.1, inplace=True)

        self.convf_2 = nn.Conv2d(1024, self.target_channel_size, 1, stride=1)
        self.bnf_2 = nn.BatchNorm2d(self.target_channel_size)
        self.reluf_2 = nn.LeakyReLU(0.1, inplace=True)

        self.poolsp = nn.MaxPool2d(2, stride=4, ceil_mode=True)




        # self.drop6 = nn.Dropout2d()
        #
        # # fc7
        #
        # self.fc7 = nn.Conv2d(4096, 4096, 1)
        # self.relu7 = nn.ReLU(inplace=True)
        # self.drop7 = nn.Dropout2d()
        #
        # self.score_fr = nn.Conv2d(4096, n_class, 1)
        # self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)
        #
        # # ***this is the end of my code

        # losses


        # if pretrained:
        #     self.load_state_dict(torch.load(yoloweights))

        self.setup_losses()




    #***MC start
    #from leimao's blog
    # def reorg(self, arrayIn, batch, C, H, W, stride, forward=False):
    #     arrayLen = len(arrayIn)
    #     arrayOut = np.zeros(arrayLen)
    #
    #     out_c = C//(stride*stride)
    #
    #     for b in range(batch):
    #         for k in range(C):
    #             for j in range(H):
    #                 for i in range(W):
    #                     in_index = i + W*(j + H*(k + C*b))
    #                     c2 = k % out_c
    #                     offset = k // out_c
    #                     w2 = i*stride + offset % stride
    #                     h2 = j*stride + offset // stride
    #                     out_index = int(w2 + W*stride*(h2 + H*stride * (c2 + out_c*b)))
    #                     if forward:
    #                         arrayOut[out_index] = arrayIn[in_index]
    #                     else:
    #                         arrayOut[in_index] = arrayIn[out_index]
    #     return arrayOut

    #https://leimao.github.io/blog/Reorg-Layer-Explained/
    def reorg(self,arrayIn, batch, C, H, W, stride, forward=False):
        arrayLen = len(arrayIn)
        arrayOut = np.zeros(arrayLen)
        out_c = C // (stride * stride)
        for b in range(batch):
            for k in range(C):
                for j in range(H):
                    for i in range(W):
                        in_index = i + W * (j + H * (k + C * b))
                        c2 = k % out_c
                        offset = k // out_c
                        w2 = i * stride + offset % stride
                        h2 = j * stride + offset // stride
                        out_index = int(w2 + W * stride * (h2 + H * stride * (c2 + out_c * b)))
                        if forward:
                            arrayOut[out_index] = arrayIn[in_index]
                        else:
                            arrayOut[in_index] = arrayIn[out_index]

        ob = batch
        oc = C * stride *stride
        oh = H // stride
        ow = W // stride

        arrayOut = np.reshape(arrayOut, (ob, oc, oh, ow))
        return arrayOut




    #*** MC End


    # *** My Code


    # def initialize_weights(self):




    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             m.weight.data.zero_()
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         if isinstance(m, nn.ConvTranspose2d):
    #             assert m.kernel_size[0] == m.kernel_size[1]
    #             initial_weight = self.get_upsampling_weight(
    #                 m.in_channels, m.out_channels, m.kernel_size[0])
    #             m.weight.data.copy_(initial_weight)
    #
    # def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
    #     """Make a 2D bilinear kernel suitable for upsampling"""
    #     factor = (kernel_size + 1) // 2
    #     if kernel_size % 2 == 1:
    #         center = factor - 1
    #     else:
    #         center = factor - 0.5
    #
    #     og = np.ogrid[:kernel_size, :kernel_size]
    #     filt = (1 - abs(og[0] - center) / factor) * \
    #            (1 - abs(og[1] - center) / factor)
    #     weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    #     weight[range(in_channels), range(out_channels), :, :] = filt
    #     return torch.from_numpy(weight).float()

    # ***MC End
    def setup_losses(self):

        self.action_ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.object_ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x):
        # print("x[0]", x[0].shape)
        #
        # img = x[0][1].cpu().numpy()
        # img = Image.fromarray(np.uint8(img))
        # plt.imshow(img)
        # plt.show()









        # split it into different types of data
        height, width = x.size()[2:]
        #print("width and height ", height, width)
        assert height == width
        assert height % 32 == 0

        target_height, target_width = int(height / 32), int(width / 32)

        #x = self.features(x)
        # View means reshape the tensor, below it's reshaping the the output tensor of self.conv(x).view(parameters)
        # original starts
        #x = self.conv(x).view(-1, self.hand_vector_size + self.object_vector_size, self.depth_discretization, target_height, target_width)

        #print("Shape of x" , x.shape)
        # End

        # *** MC start

        # +++++++++
        #print("Input shape after resnet", x.shape)
        h = x
        #print("Layer  0 :",h.shape )
        #print("Original :", torch.sum(h))
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        print("Layer 1 :", torch.sum(h[0][1]))

        #print("Layer  1 :",h.shape )
        h = self.pool1(h)
        #print(torch.sum(h))

        print("Layer P 2 :",h.shape )
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))

        print("Layer  3 :",h.shape )
        h = self.pool2(h)

        print("Layer P 4 :",h.shape )

        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))

        print("Layer  5 :",h.shape )
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))

        print("Layer  6 :",h.shape )
        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))

        print("Layer  7 :",h.shape )
        h = self.pool3(h)

        print("Layer P 8 :",h.shape )
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))

        print("Layer  9 :",h.shape )
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))

        print("Layer  10 :",h.shape )
        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))

        #print("Layer 11 :",h.shape )


        h = self.pool4(h)
        print("Layer 12 :", h.shape)


        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))

        print("Layer P 13 :", h.shape)

        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        print("Layer 14 :", h.shape)
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        print("Layer  15:", h.shape)
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        print("Layer 16 :", h.shape)
        h16 = self.relu_rout16(self.bn_rout16(self.conv_rout16(h)))
        print("Layer 17 :", h.shape)

        h = self.pool5(h16)
        print("Layer 18 :", h.shape)

        h = self.relu6_1(self.bn6_1(self.conv6_1(h)))

        #Temp start
        #hf = self.reluf_2(self.convf_2(h))

        #Temp End

        print("Layer 19 :", h.shape)
        h = self.relu6_2(self.bn6_2(self.conv6_2(h)))
        print("Layer 20 :", h.shape)
        h = self.relu6_3(self.bn6_3(self.conv6_3(h)))
        print("Layer 21 :", h.shape)
        h = self.relu7_1(self.bn7_1(self.conv7_1(h)))
        print("Layer 22 :", h.shape)
        h = self.relu7_2(self.bn7_2(self.conv7_2(h)))
        print("Layer 23 :", h.shape)
        h = self.relu7_3(self.bn7_3(self.conv7_3(h)))
        print("Layer 24 :", h.shape)
        hf = self.relu8_1(self.bn8_1(self.conv8_1(h)))
        #print("Before Route", torch.sum(hf))
        print("Layer 25 :", hf.shape)
        print("this the shape of last layer", hf.shape)


        #from here layer 25 is route to layer 16 so auther calls it route layer
        #route layer and layer 26 convolution
        h16 = self.relu_rout16(self.conv_rout26(h16))
        #print("Route :", torch.sum(h16))
        #print("this is the layer of 16 layer",type(h16))


        #from here, Reorg layer , which has been adapted from YoLo2
        # Auther calls it reorg
        b, c, h, w =h16.size()[0:]

        arrayIn = np.array(list(range( b*h*w*c))).astype(np.int)

        hr = self.reorg(arrayIn, b, c, h, w, 2, forward=False)
        # reorg returns numpy array so we nee to convert it to torch tensor
        hr = torch.from_numpy(hr).cuda().float()
        #print("Reorg", torch.sum(hr))
        #
        #print("This is reorg layer Shape :", type(hr))
        #hr = hr.to('cuda' if torch.cuda.is_available() else 'cpu')
        hcat = torch.cat((hr,hf), dim=1)
        #print("Concatenated output" , hcat.shape)

        hf = self.reluf_1(self.bnf_1(self.convf_1(hcat)))
        print("Second Last Layer :", hf.shape)
        hf = self.reluf_2(self.bnf_2(self.convf_2(hf)))

        # print("Last Layer", hf.shape)
        #
        # +++++++++
        #Important:make sure you understand below statement
        hf = hf.view(-1, self.object_vector_size + self.hand_vector_size, self.depth_discretization, target_height, target_width)

        pred_v_h = hf[:, :self.hand_vector_size, :, :, :]
        #print("Pred_v_h : ", pred_v_h.shape)
        pred_v_o = hf[:, self.hand_vector_size:, :, :, :]
        #print("Final", torch.sum(hf))
        #print("pred_v_o", pred_v_o.shape)
        # ***MC End

        # original
        # pred_v_h = x[:, :self.hand_vector_size, :, :, :]
        # pred_v_o = x[:, self.hand_vector_size:, :, :, :]
        # End of oringinal
        #print("Pred", torch.sum(hf), torch.sum(pred_v_h))
        # hand specific predictions
        pred_hand_pose = pred_v_h[:, :3 * self.num_hand_control_points, :, :, :]
        pred_hand_pose = pred_hand_pose.view(-1, 21, 3, 5, 13, 13)
        pred_hand_pose_root = torch.sigmoid(pred_hand_pose[:, 0, :, :, :, :].unsqueeze(1))
        pred_hand_pose_without_root = pred_hand_pose[:, 1:, :, :, :, :]
        pred_hand_pose = torch.cat((pred_hand_pose_root, pred_hand_pose_without_root), 1).view(-1, 63, 5, 13, 13)
        # pred_action_prob = pred_v_h[:, 3 * self.num_hand_control_points:-1, :, :, :]
        # pred_hand_conf = torch.sigmoid(pred_v_h[:, -1, :, :, :])


        #Original

        # # object specific predictions
        # pred_object_pose = pred_v_o[:, :3 * self.num_object_control_points, :, :, :]
        # #print("Pred_object_pose", pred_object_pose.shape)
        # pred_object_pose = pred_object_pose.view(-1, 21, 3, 5, 13, 13)
        # pred_object_pose_root = torch.sigmoid(pred_object_pose[:, 0, :, :, :, :].unsqueeze(1))
        # pred_object_pose_without_root = pred_object_pose[:, 1:, :, :, :, :]
        # pred_object_pose = torch.cat((pred_object_pose_root, pred_object_pose_without_root), 1).view(-1, 63, 5, 13, 13)
        # pred_object_prob = pred_v_o[:, 3 * self.num_object_control_points:-1, :, :, :]
        # pred_object_conf = torch.sigmoid(pred_v_o[:, -1, :, :, :])

        #return pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf

        #MC start
        return pred_hand_pose

    def total_loss(self, pred, true):
        #Original
        # pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = pred
        # true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = true


        #MC start
        pred_hand_pose = pred
        true_hand_pose = true

        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose, None)


        #Original
        # total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask) + self.conf_loss(
        #     pred_object_conf, pred_object_pose, true_object_pose, object_mask)
        # total_action_loss = self.action_loss(pred_action_prob, true_action_prob, hand_mask)
        # total_object_loss = self.object_loss(pred_object_prob, true_object_prob, object_mask)


       #MC start
        #total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask)

       #MC End
        # print("total pose loss", total_pose_loss)

        #Original
        #total_loss = total_pose_loss + total_conf_loss - total_action_loss - total_object_loss

        # MC starts: From here I'm editing the Total loss function so that I can get all
        total_loss = total_pose_loss

        #return total_loss, total_pose_loss, total_conf_loss, total_action_loss, total_object_loss

        # MC End
        #original
        return total_loss

    def pose_loss(self, pred, true, mask):
        print('TYpe at Pose Loss function', type(pred), type(true))
        pred = pred.view(-1, 21, 3, 5, 13, 13)
        true = true.view(-1, 21, 3, 5, 13, 13)
        #print("Pred and True and Pose loss function", torch.sum(pred), torch.sum(true))
        #original
        # masked_pose_loss = torch.mean(
        #     torch.sum(mask * torch.sum(torch.mul(pred - true, pred - true), dim=[1, 2]), dim=[1, 2, 3]))

        #MC Start

        masked_pose_loss = torch.sum(torch.mul(pred - true, pred - true))
        #MC End
        return masked_pose_loss

    def conf_loss(self, pred_conf, pred, true, mask):

        pred = pred.view(-1, 21, 3, 5, 13, 13)
        true = true.view(-1, 21, 3, 5, 13, 13)

        pred_pixel_x = pred[:, :, 0, :, :, :].unsqueeze(2) * 32 * (1920. / 416)
        pred_pixel_y = pred[:, :, 1, :, :, :].unsqueeze(2) * 32 * (1080. / 416)
        pred_depth = pred[:, :, 2, :, :, :] * 15 * 10

        true_pixel_x = true[:, :, 0, :, :, :].unsqueeze(2) * 32 * (1920. / 416)
        true_pixel_y = true[:, :, 1, :, :, :].unsqueeze(2) * 32 * (1080. / 416)
        true_depth = true[:, :, 2, :, :, :] * 15 * 10

        pixel_x_distance = torch.mul(pred_pixel_x - true_pixel_x, pred_pixel_x - true_pixel_x)
        pixel_y_distance = torch.mul(pred_pixel_y - true_pixel_y, pred_pixel_y - true_pixel_y)
        pixel_distance = torch.sqrt(pixel_x_distance + pixel_y_distance).squeeze(2)
        depth_distance = torch.sqrt(torch.mul(pred_depth - true_depth, pred_depth - true_depth))

        # threshold
        pixel_distance_mask = (pixel_distance < parameters.pixel_threshold).type(torch.cuda.FloatTensor)
        depth_distance_mask = (depth_distance < parameters.depth_threshold).type(torch.cuda.FloatTensor)
        #print("Pixel distance mask and depth, threshold 75", pixel_distance_mask.shape, depth_distance_mask.shape)
        pixel_conf = torch.exp(parameters.sharpness * (1 - pixel_distance / parameters.pixel_threshold)) / torch.exp(
            parameters.sharpness * (1 - torch.zeros(pixel_distance.size()).cuda()))
        depth_conf = torch.exp(parameters.sharpness * (1 - depth_distance / parameters.depth_threshold)) / torch.exp(
            parameters.sharpness * (1 - torch.zeros(depth_distance.size()).cuda()))

        pixel_conf = torch.mean(pixel_distance_mask * pixel_conf, dim=1)
        depth_conf = torch.mean(depth_distance_mask * depth_conf, dim=1)

        true_conf = 0.5 * (pixel_conf + depth_conf)
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)

        exist_conf_error = torch.mean(torch.sum(mask * squared_conf_error, dim=[1, 2, 3]))

        true_conf = torch.zeros(pred_conf.size()).cuda()
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        no_exist_conf_error = torch.mean(torch.sum((1 - mask) * squared_conf_error, dim=[1, 2, 3]))

        return 5 * exist_conf_error + 0.1 * no_exist_conf_error

    def action_loss(self, pred, true, mask):
        action_ce_loss = self.action_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * action_ce_loss, dim=[1, 2, 3]))

    def object_loss(self, pred, true, mask):
        object_ce_loss = self.object_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * object_ce_loss, dim=[1, 2, 3]))


if __name__ == '__main__':
    model = UnifiedNetwork()
    x = torch.randn(16, 3, 416, 416)

    true = torch.randn(32, 76, 5, 13, 13), torch.randn(32, 74, 5, 13, 13)
    # print("shape of X", x.shape)
    # print("Type of Input ",type(x))
    pred = model(x)

    true_hand_pose = torch.randn(32, 3 * parameters.num_hand_control_points, 5, 13, 13)
    true_action_prob = torch.empty(32, 5, 13, 13, dtype=torch.long).random_(parameters.num_actions)
    hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
    hand_mask[0, 0, 0] = 1.

    true_object_pose = torch.randn(32, 3 * parameters.num_object_control_points, 5, 13, 13)
    true_object_prob = torch.empty(32, 5, 13, 13, dtype=torch.long).random_(parameters.num_objects)
    object_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
    object_mask[0, 0, 0] = 1.

    true = true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask

    print(model.total_loss(pred, true))
