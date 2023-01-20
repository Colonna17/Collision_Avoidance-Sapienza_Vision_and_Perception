import torch
import torch.nn as nn

from src.utils import custom_from_numpy
from src.opticalFlowEstimator import OpticalFlowEstimator

class Classificator(nn.Module):
    def __init__(self, device, parameters):
        super().__init__()
        self.device = device
        self.loss_fn = nn.BCELoss()

        self.opticalFlowEstimator = OpticalFlowEstimator(device)

        lstm_input_size = parameters['img_height'] * parameters['img_width'] * 3 # Due to the size of a frame [ToDo: Make more robust this part]
        # print(lstm_input_size, parameters['img_width'], int((parameters['img_width'])/16)*9)
        self.lstm = torch.nn.LSTM(lstm_input_size, parameters['lstm_hidden_dim'])
        self.relu = torch.nn.ReLU()

        self.linear_classifier = torch.nn.Linear(parameters['lstm_hidden_dim'], 1)
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, curr_frame, prev_frame, detections, h=None):
        # if(some_condition): input = torch.from_numpy(input).to(device=device, dtype=torch.float32)
        curr_frame = torch.permute(custom_from_numpy(curr_frame, self.device), (2,0,1)).unsqueeze(0)
        prev_frame = torch.permute(custom_from_numpy(prev_frame, self.device), (2,0,1)).unsqueeze(0)
        
        flow = self.opticalFlowEstimator(prev_frame, curr_frame)[-1]

        input = curr_frame
        # ToDo: use CNN here to produce the input for the lstm
        flat_input = torch.flatten(input).unsqueeze(0)

        output, (h, c) = self.lstm(flat_input, h)
        output = self.relu(output)

        output = self.linear_classifier(output)
        output = self.sigmoid(output)

        return output, (h, c)
    
    def loss(self, pred, y):
        return self.loss_fn(pred, y)

