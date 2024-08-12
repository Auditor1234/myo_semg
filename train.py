import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm
from window_dataloader import CNNDataset
from common_utils import save_model_weight
from loss import combine_loss
from utils import ACC, plot_features, plot_uncertainty_accuracy



def train(model, epochs, train_X, train_y, val_X, val_y, test_X, test_y, 
          loss_func=nn.CrossEntropyLoss(), 
          subject=0, 
          file=None, 
          weight_path='res/best.pt',
          device=None,
          dist_path=None):
    train_X = torch.tensor(train_X, dtype=torch.float16)
    train_y = torch.tensor(train_y)
    val_X = torch.tensor(val_X, dtype=torch.float16)
    val_y = torch.tensor(val_y)
    test_X = torch.tensor(test_X, dtype=torch.float16)
    test_y = torch.tensor(test_y)

    train_loader = DataLoader(
        dataset=CNNDataset(train_X, train_y),
        batch_size=32,
        # drop_last=True
        )

    eval_loader = DataLoader(
        dataset=CNNDataset(val_X, val_y),
        batch_size=32,
        # drop_last=True
        )
    
    test_loader = DataLoader(
        dataset=CNNDataset(test_X, test_y),
        batch_size=32,
        # drop_last=True
        )

    num_class = len(torch.unique(test_y))
    if isinstance(loss_func, nn.Module):
        loss_func.to(device)
    model = model.to(device).half()
    optimizer = Adam(model.parameters(), lr=0.001, eps=1e-2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = None

    current_precision, best_precision = 0, 0
    loss, acc = [], []
    print('Start training...')
    for epoch in range(epochs):
        if isinstance(loss_func, nn.Module):
            train_loss = train_one_epoch_fuse(model, epoch, epochs, device, train_loader, loss_func, optimizer, scheduler, dist_path=None)
        else:
            train_loss = train_one_epoch(model, epoch, epochs, device, train_loader, loss_func, optimizer, scheduler, dist_path=None)
        loss.append(train_loss)
        current_precision, _, _ = validate(model, epoch, device, eval_loader, num_class)
        acc.append(current_precision)

        if current_precision >= best_precision:
            best_precision = current_precision
            save_model_weight(model=model, filename=weight_path)
    
    print('\nCurrent best precision in val set is: %.4f' % (best_precision * 100) + '%')
    model.load_state_dict(torch.load(weight_path))
    acc, region_acc, split_acc = validate(model, epoch, device, test_loader, num_class, subject, file, dist_path=dist_path)
    # validate(model, epoch, device, eval_loader, num_class, subject, file='res/img/12_s3_3expert_val.png')

    return acc, region_acc, split_acc


def train_one_epoch(model, epoch, epochs, device, train_loader, loss_func, optimizer, scheduler, dist_path):
    model.train()
    x_feat = []
    y_feat = []
    weights = []
    total_loss = 0.0
    loop = tqdm(train_loader, desc='Train')
    for _, (x_data, y_label) in enumerate(loop): # shape(B, 12, 200, 1)

        x_data = x_data.to(device)
        
        predict, _, _ = model(x_data)
        x_feat.append(predict)
        y_feat.append(y_label)

        labels = y_label.type(torch.long).to(device)
        loss = loss_func(predict, labels, epoch)
        total_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loop.set_postfix(loss = loss.item())

    if epoch == epochs - 1 and dist_path:
        x_feat = torch.cat(x_feat, dim=0).cpu().detach().numpy()
        y_feat = torch.cat(y_feat, dim=0).cpu().detach().numpy()
        plot_features(x_feat, y_feat, dist_path)

    if scheduler != None:
        scheduler.step()
    print("[%d/%d] epoch's average loss = %.6f" % (epoch + 1, epochs, total_loss / len(train_loader)))
    return total_loss.item() / len(train_loader)


def train_one_epoch_fuse(model, epoch, epochs, device, train_loader, loss_func, optimizer, scheduler, dist_path):
    model.train()
    x_feat = []
    y_feat = []
    weights = []
    loss_func._hook_before_epoch(epoch)
    total_loss = 0.0
    loop = tqdm(train_loader, desc='Train')
    for _, (x_data, y_label) in enumerate(loop): # shape(B, 12, 200, 1)

        x_data = x_data.to(device)
        labels = y_label.type(torch.long).to(device)
        predict, feat, weights = model(x_data)
        x_feat.append(feat)
        y_feat.append(y_label)
        extra_info = {
            "num_expert"    : len(model.logits),
            "logits"        : model.logits,
            "w"             : model.w,
            "feat"          : feat,
            "weights"       : weights
        }

        loss = loss_func(predict, labels, epoch, extra_info)
        total_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loop.set_postfix(loss = loss.item())
    if epoch == epochs - 1 and dist_path:
        x_feat = torch.cat(x_feat, dim=0).cpu().detach().numpy()
        y_feat = torch.cat(y_feat, dim=0).cpu().detach().numpy()
        plot_features(x_feat, y_feat, dist_path)
    if scheduler != None:
        scheduler.step()
    print("[%d/%d] epoch's average loss = %.6f" % (epoch + 1, epochs, total_loss / len(train_loader)))
    return total_loss.item() / len(train_loader)


def validate(model, epoch, device, val_loader, num_class, subject=0, file=None, dist_path=None):
    model.eval()
    model._hook_before_epoch(epoch)
    output = torch.empty(0, num_class, dtype=torch.float32, device=device)
    targets = []
    x_feat = []
    y_feat = []

    print("Validating...")
    loop = tqdm(val_loader, desc='Validation')
    for i, (x_data, y_label) in enumerate(loop): # shape(4, *, 12)
        predict, feat, _ = model(x_data.to(device), y_label)
        x_feat.append(feat)
        y_feat.append(y_label)

        output = torch.cat([output, predict.detach()],dim=0)
        targets.append(y_label)
        loop.set_description(f'Validation [{epoch + 1}]')
    targets = torch.cat(targets).to(device)
    model._hook_after_epoch(epoch)
    acc, region_acc, split_acc = ACC(output, targets, region_len=num_class / 3, subject=subject, file=file)
    if epoch == 60 - 1 and file:
        plot_uncertainty_accuracy(output, targets)
    
    if epoch == 60 - 1 and dist_path:
        x_feat = torch.cat(x_feat, dim=0).cpu().detach().numpy()
        y_feat = torch.cat(y_feat, dim=0).cpu().detach().numpy()
        plot_features(x_feat, y_feat, dist_path)


    return acc, region_acc, split_acc
