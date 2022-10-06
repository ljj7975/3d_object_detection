import torch.nn.functional as F

def train(
    model,
    epoch,
    train_loader,
    optimizer,
    writer = None,
):
    """Train single epoch"""
    model.train()
    num_train_samples = len(train_loader.dataset)
    log_interval = len(train_loader) // 5

    for batch_idx, (points, target) in enumerate(train_loader):
        optimizer.zero_grad()
        points = points.cuda()
        target = target.cuda()
        points = points.transpose(2, 1) # points.shape needs to be (num_batch, 3 (num_dims), num_pts)
        output, _, _ = model(points)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(points)}/{num_train_samples} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data.item():.6f}")

            # Log train/loss to TensorBoard
            n_iter = (epoch - 1) * num_train_samples + batch_idx * len(points)
            if writer is not None:
                writer.add_scalar('train/loss', loss.data.item(), n_iter)


# evaluate model
def evaluate(
    model,
    dataset_split,
    data_loader,
    writer=None,
    n_iter = None
):
    """Evaluate the given model"""
    model.eval()
    test_loss = 0
    correct = 0

    for points, target in data_loader:
        points = points.cuda()
        target = target.cuda()
        points = points.transpose(2, 1)
        output, _, _ = model(points)
        test_loss += F.nll_loss(output, target, size_average=False).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    num_samples = len(data_loader.dataset)
    test_loss /= num_samples
    test_accuracy = 100. * correct / num_samples
    print(f"{dataset_split.value} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{num_samples} ({test_accuracy:.2f}%)\n")

    # Log val/loss & val/accuracy
    if n_iter is not None and writer is not None:
        writer.add_scalar(f'{dataset_split.value}/loss', test_loss, n_iter)
        writer.add_scalar(f'{dataset_split.value}/accuracy', test_accuracy, n_iter)

