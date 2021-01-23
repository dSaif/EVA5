from tqdm import tqdm
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc, l1_loss_flag, lamda_l1=0.0001):

  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)

    if l1_loss_flag:
       l1 = 0
       for p in model.parameters():
         l1 = l1+p.abs().sum()
       loss += lamda_l1 * l1

    train_losses.append(loss)  # .item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    # get the index of the max log-probability
    pred = y_pred.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(
        desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)