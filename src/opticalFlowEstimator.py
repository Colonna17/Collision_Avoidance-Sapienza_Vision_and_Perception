
# import torch
import torch.nn as nn

from torchvision.models.optical_flow import raft_small
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

class OpticalFlowEstimator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms() # Not used yet
        self.estimator = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)

        for param in self.parameters():
            param.requires_grad = False # For the moment, we don't want to train this model.

    def forward(self, prev_frame, curr_frame):
        return self.estimator(prev_frame, curr_frame)

    @staticmethod
    def to_image(optical_flow):
        optical_flow = flow_to_image(optical_flow).squeeze(0)
        return F.to_pil_image(optical_flow.to("cpu"))
