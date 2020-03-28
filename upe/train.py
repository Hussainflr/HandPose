import random

from tqdm import tqdm
import torch
import numpy as np
from upe.cfg import parameters
from upe.jpunet_o import UnifiedNetwork
from upe.dataset_o import UnifiedPoseDataset

from tensorboardX import SummaryWriter
from torch.utils.data import Sampler
import torchvision

from matplotlib import pyplot as plt
from upe.imageloader import visualize_joints_2d




class WithReplacementRandomSampler(Sampler):
    """Samples elements randomly, with replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # generate samples of `len(data_source)` that are of value from `0` to `len(data_source)-1`
        samples = torch.LongTensor(len(self.data_source))
        samples.random_(0, len(self.data_source))
        return iter(samples)

    def __len__(self):
        return len(self.data_source)



def main():
    training_dataset = UnifiedPoseDataset(mode='train', loadit=True, name='train2')
    testing_dataset = UnifiedPoseDataset(mode='test', loadit=True, name='test2')
    samples = []
    samples += training_dataset.samples + testing_dataset.samples
    print(samples[0])
    random.shuffle(samples)
    print(samples[0])
    training_dataset.samples = samples[:int(len(samples) / 2)]
    testing_dataset.samples = samples[int(len(samples) / 2):]

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=16, sampler=WithReplacementRandomSampler(training_dataset), shuffle=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, shuffle=False)

    model = UnifiedNetwork()

    model.initialize_weights()

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr, weight_decay=parameters.weight_decay)

    best_loss = float('inf')

    writer = SummaryWriter()
    #print("Weights before backprop :", list(model.parameters()))
    for epoch in range(parameters.epochs):
        running_loss = 0.0
        if epoch == 80 or epoch == 160:
            for g in optimizer.param_groups:
                #print("Current LR ", g['lr'])
                g['lr'] = g['lr']/10
                #print("Updated lr", g['lr'])
        # train

        model.train()
        training_loss = 0.0
        # print("Number of samples", len(training_dataloader))
        for batch, data in enumerate(tqdm(training_dataloader)):
            optimizer.zero_grad()
            #print("batach size", batch.__sizeof__())
            image = data[0]
            #show_Image_Pose(data, training_dataset)

            #original
            #true = [x.cuda() for x in data[1:]]

            #MC

            true = data[1].cuda()
            #MC End
            #print("Image shape from train", type(image.cuda()))
            #pred = model(image.cuda())
            pred = model(image.cuda())
            #print('type at training', pred.shape)
            #print("Prediction shape :", len(pred))

            print("At training", pred.shape)
            loss = model.total_loss(pred, true)

            running_loss += loss.item()
            print("Running loss : ", running_loss)
            # if batch % 567 == 566:
            #     print("Running loss : ", running_loss)

            training_loss += loss.cpu().data.numpy()

            #training_loss += loss.item()
            loss.backward()
            #print("Trainging loss", training_loss)

            optimizer.step()


        training_loss = training_loss / batch
        #print(" Training loss per Batch", training_loss)
        writer.add_scalars('data/loss', {'train_loss': training_loss}, epoch)

        # validation
        # model.eval()
        validation_loss = 0.
        with torch.no_grad():
            for batch, data in enumerate(tqdm(testing_dataloader)):
                image = data[0]
                true = [x.cuda() for x in data[1:]]

                #pred = model(image.cuda())
               # print("Type of input",  type(image.cuda))

                #print("Input Image type", type(image))
                pred = model(image.cuda())
                loss = model.total_loss(pred, true)
                validation_loss += loss.data.cpu().numpy()

        validation_loss = validation_loss / batch
        writer.add_scalars('data/loss', {'val_loss': validation_loss}, epoch)

        if validation_loss < best_loss:
            print("Old loss: {}, New loss : {}. Saving model to disk.".format(best_loss, validation_loss))
            best_loss = validation_loss

            torch.save(model.state_dict(), '../models/unified_net.pth')

        print(
            "Epoch : {} finished. Training Loss: {}. Validation Loss: {}".format(epoch, training_loss, validation_loss))

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()




def show_Image_Pose(data, training_dataset):
    # print("Data size at train. ", type(data[1].argmin()))
    #
    # print("Data size at train. 2 ", data[2].shape)
    #
    # print("hand Mask shape at train. 3", data[3].shape)
    hand_pose = data[1]
    hand_mask = data[3]
    hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)

    z, v, u = hand_cell[1:]

    dels = hand_pose[0, :, z, v, u]

    dels = dels.reshape(21, 3).numpy()

    #print("dels shape :", type(dels))
    delz, delv, delu = dels[:, 0], dels[:, 1], dels[:, 2]

    controlpoints = training_dataset.target_to_control(delz, delv, delu, (z, v, u))
    image = data[0]
    for i in range(image.size(0)):
        torchvision.utils.save_image(image[i, :, :, :],'{}.png'.format(i))


    #fig = plt.figure()

    #ax = fig.add_subplot(221)

    #print(image)
    #lets reshape the Tensor into (W, H, C) so that we can show it. Note original shape of tensor is (C, W, H)

    #image =  image.view(image.shape[1], image.shape[2], image.shape[0]).numpy()
    #print(image.shape)
    #plt.imshow(image.permute(1, 2, 0))
    #print("shape of control points", controlpoints.shape)

    #visualize_joints_2d(ax, controlpoints)
    #plt.show()




if __name__ == '__main__':
    main()
