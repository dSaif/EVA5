This a model for MNIST dataset.
The model uses Pytorch's conv2d, batch normalization, dropout, adaptive average pooling, relu and softmax.

Model achives it's best accuracy 99.41 of in 11th epoch. 

Architecture of Model:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1            [-1, 8, 26, 26]              80
       BatchNorm2d-2            [-1, 8, 26, 26]              16
           Dropout-3            [-1, 8, 26, 26]               0
            Conv2d-4           [-1, 16, 24, 24]           1,168
       BatchNorm2d-5           [-1, 16, 24, 24]              32
           Dropout-6           [-1, 16, 24, 24]               0
         MaxPool2d-7           [-1, 16, 12, 12]               0
            Conv2d-8           [-1, 32, 10, 10]           4,640
       BatchNorm2d-9           [-1, 32, 10, 10]              64
          Dropout-10           [-1, 32, 10, 10]               0
           Conv2d-11             [-1, 32, 8, 8]           9,248
      BatchNorm2d-12             [-1, 32, 8, 8]              64
          Dropout-13             [-1, 32, 8, 8]               0
           Conv2d-14             [-1, 16, 8, 8]             528, relu 
      BatchNorm2d-15             [-1, 16, 8, 8]              32
          Dropout-16             [-1, 16, 8, 8]               0
           Conv2d-17             [-1, 16, 6, 6]           2,320
      BatchNorm2d-18             [-1, 16, 6, 6]              32
           Conv2d-19             [-1, 10, 4, 4]           1,450
        AdaptiveAvgPool2d-20             [-1, 10, 1, 1]               0
----------------------------------------------------------------
Total params: 19,674
Trainable params: 19,674
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.51
Params size (MB): 0.08
Estimated Total Size (MB): 0.58
----------------------------------------------------------------

Training Log of the model:
  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:65: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
epoch=1 loss=0.05661095678806305 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.22it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0569, Accuracy: 9825/10000 (98.25%)

epoch=2 loss=0.028985721990466118 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.69it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0360, Accuracy: 9891/10000 (98.91%)

epoch=3 loss=0.008123626001179218 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.26it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0305, Accuracy: 9904/10000 (99.04%)

epoch=4 loss=0.01380994077771902 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.03it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0292, Accuracy: 9911/10000 (99.11%)

epoch=5 loss=0.02616730146110058 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.46it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0281, Accuracy: 9907/10000 (99.07%)

epoch=6 loss=0.004758123774081469 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.68it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0249, Accuracy: 9929/10000 (99.29%)

epoch=7 loss=0.006347087677568197 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.68it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0239, Accuracy: 9926/10000 (99.26%)

epoch=8 loss=0.025747528299689293 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.85it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0232, Accuracy: 9931/10000 (99.31%)

epoch=9 loss=0.04350179433822632 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.92it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0242, Accuracy: 9927/10000 (99.27%)

epoch=10 loss=0.019045144319534302 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.59it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0210, Accuracy: 9941/10000 (99.41%)

epoch=11 loss=0.023883521556854248 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.45it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0252, Accuracy: 9922/10000 (99.22%)

epoch=12 loss=0.002942350460216403 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 33.09it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0231, Accuracy: 9934/10000 (99.34%)

epoch=13 loss=0.004819503054022789 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.87it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0218, Accuracy: 9937/10000 (99.37%)

epoch=14 loss=0.04620729386806488 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 32.84it/s]
Test set: Average loss: 0.0226, Accuracy: 9935/10000 (99.35%)


