import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Qnetwork(nn.Module):
    def __init__(self, lr, n_actions, input_shape):
        super(Qnetwork, self).__init__()

        self.cv1 = nn.Conv2d(input_shape[0], 32, 8, stride=4)
        self.cv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.cv3 = nn.Conv2d(64, 64, 3, stride=1)

        cv3_output_shape = self.get_cv3_output_shape(input_shape)

        self.fc1 = nn.Linear(cv3_output_shape, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def get_cv3_output_shape(self, input_shape):
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)

        return convw * convh * 64

    def forward(self, state):
        cv1 = F.relu(self.cv1(state))
        cv2 = F.relu(self.cv2(cv1))
        cv3 = F.relu(self.cv3(cv2))
        # cv3 shape is BS x n_filters x H x W
        conv_state = cv3.view(cv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1= F.relu(self.fc1(conv_state))
        V = self.V(flat1)
        A = self.A(flat1)

        actions = V + A - A.mean()


        return actions