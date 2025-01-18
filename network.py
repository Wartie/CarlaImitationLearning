import torch
import torch.nn as nn
from operator import add
from torchsummary import summary
import torch.nn.functional as F


class ClassificationNetwork(torch.nn.Module):
    def __init__(self, use_observations):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        self.width = 320
        self.height = 240

        self.behavior_to_key_map = {0: [0.5, 0.0, 0.0], #gas
                                    1: [0.0, -0.35, 0.0], #steer left
                                    2: [0.0, 0.35, 0.0], #steer right
                                    3: [0.0, 0.0, 0.3], #brake
                                    4: [0.0, 0.0, 0.0] #do nothing
                                    }

        self.behavior_map = {0: [0],
                             1: [1],
                             2: [2],
                             3: [3], 
                             4: [0, 1],
                             5: [0, 2],
                             6: [3, 1],
                             7: [3, 2],
                             8: [4]}
                            #  9: [4, 0, 1],
                            # 10: [4, 0, 2],
                            # 11: [-1]}
        self.use_other_observations = use_observations
        if not self.use_other_observations:
            self.model = nn. Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), #input: 320 x 240 -> 160 * 120 * 32
                nn.ReLU(),
                nn.MaxPool2d(5, stride=2, padding=2), #will cut down to 80 * 60 * 32
                nn.BatchNorm2d (16),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), #results in 80 * 60 * 64
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1), #will cut down to 40 * 30 * 64
                nn.BatchNorm2d (32),
                nn.Flatten(),
                nn.Linear(int(32 * self.width * self.height / ((2 * 2 * 2 * 2)**2)), 1920),
                nn.ReLU(),
                nn.Linear(1920, 192),
                nn.ReLU(),
                nn.Linear(192, 9),
                nn.ReLU(),
                nn.Softmax()
            )
        else:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
            self.pool1 = nn.MaxPool2d(5, stride=2, padding=2)
            self.bn1   = nn.BatchNorm2d (16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
            self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
            self.bn2   = nn.BatchNorm2d (32)

            self.fc1 = nn.Linear(int(32 * self.width * self.height / ((2 * 2 * 2 * 2)**2)) + 8, 1920)
            self.fc2 = nn.Linear(1920, 192)
            self.fc3 = nn.Linear(192, 9)

        # summary(self.model, (3, 240, 320))


    def forward(self, observation, other_observations):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        if not self.use_other_observations:
            logits = self.model(observation)
        else:
            x = self.bn1(self.pool1(F.relu(self.conv1(observation))))
            x = self.bn2(self.pool2(F.relu(self.conv2(x))))
            x = torch.flatten(x, 1) # flatten all dimensions except

            fused = torch.cat([x, other_observations], dim=1)

            x = F.relu(self.fc1(fused))
            x = F.relu(self.fc2(x))
            logits = F.softmax(self.fc3(x))
        # print(logits)
        translated = []
        for out in list(logits):
            translated.append(self.scores_to_action(list(out)))
        
        actions = torch.FloatTensor(translated).cuda()
        # print(actions.size())
        return actions
    
    def actions_to_classes(self, actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are C different classes, then
        every action is represented by a C-dim vector which has exactly one
        non-zero entry (one-hot encoding). That index corresponds to the class
        number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size C
        """
        one_hots = []
        for action in actions:
            mapping = [0, 0, 0]
            if action[0] >= 0.1:
                mapping[0] = 1
            else:
                mapping[0] = 0
            if action[1] >= 0.1:
                mapping[1] = 1
            elif action[1] <= -0.1:
                mapping[1] = -1
            else:
                mapping[1] = 0
            if action[2] >= 0.1:
                mapping[2] = 1
            else:
                mapping[2] = 0
        
            oneHot = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            if mapping[0] == 1 and mapping[1] == 0 and mapping[2] == 0:
                oneHot[0] = 1
            elif mapping[0] == 0 and mapping[1] == -1 and mapping[2] == 0:
                oneHot[1] = 1
            elif mapping[0] == 0 and mapping[1] == 1 and mapping[2] == 0:
                oneHot[2] = 1
            elif mapping[0] == 0 and mapping[1] == 0 and mapping[2] == 1:
                oneHot[3] = 1
            elif mapping[0] == 1 and mapping[1] == -1 and mapping[2] == 0:
                oneHot[4] = 1
            elif mapping[0] == 1 and mapping[1] == 1 and mapping[2] == 0:
                oneHot[5] = 1
            elif mapping[0] == 0 and mapping[1] == -1 and mapping[2] == 1:
                oneHot[6] = 1
            elif mapping[0] == 0 and mapping[1] == 1 and mapping[2] == 1:
                oneHot[7] = 1
            elif mapping[0] == 0 and mapping[1] == 0 and mapping[2] == 0:
                oneHot[8] = 1
            one_hots.append(oneHot)
        
        return one_hots


    def scores_to_action(self, scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [accelaration, steering, braking].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        actionsVals = [0.0, 0.0, 0.0]
        
        for action in scores:
            actionClass = self.behavior_map[torch.argmax(action).item()]
            for behavior in actionClass:
                map(add, actionsVals, self.behavior_to_key_map[behavior])
        return tuple(actionsVals)

class RegressionNetwork(torch.nn.Module):
    def __init__(self):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        self.width = 320
        self.height = 240

        self.model = nn. Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), #input: 320 x 240 -> 160 * 120 * 32
            nn.ReLU(),
            nn.MaxPool2d(5, stride=2, padding=2), #will cut down to 80 * 60 * 32
            nn.BatchNorm2d (32),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), #results in 80 * 60 * 64
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1), #will cut down to 40 * 30 * 64
            nn.BatchNorm2d (64),
            nn.Flatten(),
            nn.Linear(int(64 * self.width * self.height / ((2 * 2 * 2 * 2)**2)), 1920),
            nn.ReLU(),
            nn.Linear(1920, 192),
            nn.ReLU(),
            nn.Linear(192, 3),
            nn.ReLU()
        )

    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        logits = self.model(observation)
        return logits