This a model for MNIST dataset. The model uses Pytorch's conv2d, batch normalization, dropout, adaptive average pooling, relu and softmax.

Model achives it's best accuracy 99.42% of in 15th epoch.

Architecture of Model: Layer (type) Output Shape Param #

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 16, 26, 26]             160
       BatchNorm2d-2           [-1, 16, 26, 26]              32
           Dropout-3           [-1, 16, 26, 26]               0
            Conv2d-4           [-1, 16, 24, 24]           2,320
       BatchNorm2d-5           [-1, 16, 24, 24]              32
           Dropout-6           [-1, 16, 24, 24]               0
         MaxPool2d-7           [-1, 16, 12, 12]               0
            Conv2d-8           [-1, 32, 10, 10]           4,640
       BatchNorm2d-9           [-1, 32, 10, 10]              64
          Dropout-10           [-1, 32, 10, 10]               0
           Conv2d-11             [-1, 32, 8, 8]           9,248
      BatchNorm2d-12             [-1, 32, 8, 8]              64
          Dropout-13             [-1, 32, 8, 8]               0
           Conv2d-14             [-1, 16, 8, 8]             528
      BatchNorm2d-15             [-1, 16, 8, 8]              32
          Dropout-16             [-1, 16, 8, 8]               0
           Conv2d-17             [-1, 16, 6, 6]           2,320
      BatchNorm2d-18             [-1, 16, 6, 6]              32
          Dropout-19             [-1, 16, 6, 6]               0
           Conv2d-20             [-1, 10, 4, 4]           1,450
      AdaptiveAvgPool2d-21             [-1, 10, 1, 1]               0
----------------------------------------------------------------
Total params: 20,922
Trainable params: 20,922
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.63
Params size (MB): 0.08
Estimated Total Size (MB): 0.72

Training Log of the model: 


  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:66: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
epoch=1 loss=0.18028521537780762 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.21it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.1289, Accuracy: 9594/10000 (95.94%)

epoch=2 loss=0.03857574239373207 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.35it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0450, Accuracy: 9862/10000 (98.62%)

epoch=3 loss=0.03239254280924797 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.76it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0459, Accuracy: 9863/10000 (98.63%)

epoch=4 loss=0.05607930198311806 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.10it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0334, Accuracy: 9892/10000 (98.92%)

epoch=5 loss=0.010587397962808609 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.06it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0317, Accuracy: 9905/10000 (99.05%)

epoch=6 loss=0.05358893796801567 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.93it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0272, Accuracy: 9917/10000 (99.17%)

epoch=7 loss=0.013281588442623615 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.45it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0286, Accuracy: 9912/10000 (99.12%)

epoch=8 loss=0.04807808995246887 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.56it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0252, Accuracy: 9921/10000 (99.21%)

epoch=9 loss=0.025806806981563568 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.29it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0204, Accuracy: 9929/10000 (99.29%)

epoch=10 loss=0.005063475575298071 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.74it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0215, Accuracy: 9941/10000 (99.41%)

epoch=11 loss=0.08403293043375015 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.15it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0201, Accuracy: 9940/10000 (99.40%)

epoch=12 loss=0.03643565997481346 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.92it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0181, Accuracy: 9937/10000 (99.37%)

epoch=13 loss=0.0042666238732635975 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.46it/s]
  0%|          | 0/469 [00:00<?, ?it/s]

Test set: Average loss: 0.0234, Accuracy: 9928/10000 (99.28%)

epoch=14 loss=0.030948318541049957 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.02it/s]

Test set: Average loss: 0.0202, Accuracy: 9942/10000 (99.42%)
