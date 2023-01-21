import torch
import torch.nn as nn

from src.utils import custom_from_numpy
from src.opticalFlowEstimator import OpticalFlowEstimator

class Classifier(nn.Module):
    def __init__(self, device, parameters):
        super().__init__()
        self.device = device
        self.loss_fn = nn.BCELoss()
        self.max_det = parameters['max_det']

        self.opticalFlowEstimator = OpticalFlowEstimator(device)

        lstm_input_size = parameters['img_height'] * parameters['img_width']*5 + self.max_det*7 #  Frame+Flow + Tracking
        # print(lstm_input_size, parameters['img_width'], int((parameters['img_width'])/16)*9)
        self.lstm = torch.nn.LSTM(lstm_input_size, parameters['lstm_hidden_dim'], device=device)
        self.relu = torch.nn.ReLU()

        self.linear_classifier = torch.nn.Linear(parameters['lstm_hidden_dim'], 1)
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, curr_frame, prev_frame, detections, h=None):
        # if(some_condition): input = torch.from_numpy(input).to(device=device, dtype=torch.float32)
        curr_frame = torch.permute(custom_from_numpy(curr_frame, self.device), (2,0,1)).unsqueeze(0)
        prev_frame = torch.permute(custom_from_numpy(prev_frame, self.device), (2,0,1)).unsqueeze(0)
        detections = custom_from_numpy(detections, self.device)
        flow = self.opticalFlowEstimator(prev_frame, curr_frame)[-1]
        
        curr_frame = torch.flatten(curr_frame)
        flow = torch.flatten(flow)
        padding_size = (self.max_det - len(detections)) * 7
        padding = torch.ones(padding_size) * -1
        detections = (torch.flatten(detections))
        
        input = torch.cat((curr_frame, flow, detections, padding)).unsqueeze(0)
        output, (h, c) = self.lstm(input, h)
        output = self.relu(output)

        output = self.linear_classifier(output)
        output = self.sigmoid(output)

        return output, (h, c)
    
    
    def loss(self, pred, y):
        return self.loss_fn(pred, y)

