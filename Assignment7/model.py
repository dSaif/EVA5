import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(
            num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(
            num_features * self.num_splits))

    def train(self, mode=True):
        # lazily collate stats when we are going to use them
        if (self.training is True) and (mode is False):
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H,
                           W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(
                    self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


class Net(nn.Module):
    def __init__(self, is_GBN=False, gbn_splits=2, in_c = 1):
        super(Net, self).__init__()
        
        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            GhostBatchNorm(64, gbn_splits) if is_GBN else nn.BatchNorm2d(64),
            
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            GhostBatchNorm(64, gbn_splits) if is_GBN else nn.BatchNorm2d(64),
            
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            GhostBatchNorm(64, gbn_splits) if is_GBN else nn.BatchNorm2d(64)
        )  # input_size = 28 output_size = 26 receptive_field = 3

        # TRANSITION BLOCK 1
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2, 2))# input_size = 24 output_size = 12 receptive_field =

        # CONVOLUTION BLOCK 2 DSConnv
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            GhostBatchNorm(64, gbn_splits) if is_GBN else nn.BatchNorm2d(64),
            
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            GhostBatchNorm(64, gbn_splits) if is_GBN else nn.BatchNorm2d(64),
            
            
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            GhostBatchNorm(64, gbn_splits) if is_GBN else nn.BatchNorm2d(64)
        )  # input_size = 26 output_size = 24 receptive_field = 5

        # TRANSITION BLOCK 2
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2, 2))  # input_size = 24 output_size = 12 receptive_field =

        # CONVOLUTION BLOCK 3 Dilated
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            GhostBatchNorm(128, gbn_splits) if is_GBN else nn.BatchNorm2d(128),
            
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False,dilation=2),
            nn.ReLU(),
            GhostBatchNorm(128, gbn_splits) if is_GBN else nn.BatchNorm2d(128),
            
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False,dilation=2),
            nn.ReLU(),
            GhostBatchNorm(128, gbn_splits) if is_GBN else nn.BatchNorm2d(128)
        )  # input_size = 12 output_size = 10 receptive_field = 5

        # TRANSITION BLOCK 3
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(2, 2))  # input_size = 24 output_size = 12 receptive_field =

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            GhostBatchNorm(256, gbn_splits) if is_GBN else nn.BatchNorm2d(256),
            
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(1, 1), padding=1, bias=False,groups=64),
            nn.ReLU(),
            GhostBatchNorm(256, gbn_splits) if is_GBN else nn.BatchNorm2d(256),
            
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(1, 1), padding=1, bias=False),
            nn.ReLU(),
            GhostBatchNorm(256, gbn_splits) if is_GBN else nn.BatchNorm2d(256)
        )  # input_size = 12 output_size = 10 receptive_field = 5

        
        # OUTPUT BLOCK with GAP
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=256, out_channels=10,
                      kernel_size=(1, 1), padding=0, bias=False)
        )  # input_size = 5 output_size = 1  receptive_field = 29

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool2(x)
        x = self.convblock3(x)
        x = self.pool3(x)
        x = self.convblock4(x)
        x = self.output(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def get_optimizer(model_parameters,loss_type):
    if loss_type == "L2":
        optimizer = optim.SGD(model_parameters, lr=0.01,
                  momentum=0, weight_decay=0, nesterov=False)
    else:
        optimizer = optim.SGD(model_parameters, lr= 0.01, momentum=0.9)
    return optimizer


def run_model(model, device, optimiser, EPOCHS=1, is_L1_loss=False, is_GBN=False, gbn_splits=2):

  train_losses = []
  train_acc = []
  test_losses = []
  test_acc = []

  for epoch in range(EPOCHS):
      print("EPOCH:", epoch+1)
      train(model, device, train_loader, optimizer, epoch,
            train_losses, train_acc, is_L1_loss, lamda_l1=0.0001)
      test(model, device, test_loader, test_losses, test_acc)

  return {'train_loss': train_losses,  'train_acc': train_acc,  'test_loss': test_losses,  'test_acc': test_acc}
