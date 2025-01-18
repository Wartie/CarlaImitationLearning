import glob

import numpy as np

import torch
import json

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CarlaDataset(Dataset):
    def __init__(self, data_dir, is_validation, use_observation):
        self.data_dir = data_dir

        if not is_validation:
            self.data_images_list = glob.glob(self.data_dir + 'actual_data/images/' + '*.png')
            self.data_label_list = glob.glob(self.data_dir + 'actual_data/labels/' + '*.json') #need to change to your data format
        else:
            self.data_images_list = glob.glob(self.data_dir + 'validation/images/' + '*.png')
            self.data_label_list = glob.glob(self.data_dir + 'validation/labels/' + '*.json') #need to change to your data format

        self.use_observation = use_observation

        justFileNamesImages = [string[-32:-4] for string in self.data_images_list]
        justFileNamesLabels = [string[-40:-12] for string in self.data_label_list]

        new_list = list(set(justFileNamesImages).difference(justFileNamesLabels))
        print(new_list)

        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return len(self.data_images_list)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """

        curImagePath = self.data_images_list[idx]
        curLabelPath = self.data_label_list[idx]
        # print(curImagePath)

        img = Image.open(curImagePath)
        img.load()
        img_rgb = img.convert('RGB')
        imgData = np.array( img_rgb )
        # if (np.isnan(imgData).any()):
        #     print(curImagePath)

        # print(imgData.shape) #comes in as h, w, c

        action = [0.0, 0.0, 0.0]
        observation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with open(curLabelPath) as f:
            allLabel = json.load(f)
            # print(allLabel)
            # print(allLabel["Throttle:"][0], type(allLabel["Throttle:"]))
            action[0] = float(allLabel["Throttle:"][0]) 
            action[1] = float(allLabel["Steer:"][0])
            action[2] = float(allLabel["Brake:"][0])

            if self.use_observation:
                accel = [float(token.strip(" ()")) for token in allLabel["Accelero"][0].split(",")]
                gyro = [float(token.strip(" ()")) for token in allLabel["Gyroscop"][0].split(",")]
                gnss = [float(token.strip(" ()")) for token in allLabel["GNSS"][0].split(",")]
                observation[0] = float(accel[0])
                observation[1] = float(accel[1])
                observation[2] = float(accel[2])
                observation[3] = float(gyro[0])
                observation[4] = float(gyro[1])
                observation[5] = float(gyro[2])
                observation[6] = float(gnss[0])
                observation[7] = float(gnss[1])


        # print(action)
        imgAsTensor = self.transform(imgData)
        otherobsAsTensor = torch.FloatTensor(np.asarray(observation))
        actionAsTensor = torch.FloatTensor(np.asarray(action))
        
        return ((imgAsTensor, otherobsAsTensor), actionAsTensor)

def get_dataloader(data_dir="CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/homeworks/hw1/data/", batch_size=1, is_validation = False, use_observation = False, num_workers=4, shuffle=True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir, is_validation, use_observation),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )
    

