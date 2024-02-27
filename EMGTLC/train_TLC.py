import torch
from EMGTLC.metric_TLC import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from window_dataloader import CNNDataset
from torch.cuda.amp import autocast
from EMGTLC.loss_TLC import TLCLoss
from torch.optim import Adam, lr_scheduler
from common_utils import save_model_weight
import numpy as np



def train(model, epochs, train_X, train_y, val_X, val_y, test_X, test_y):
    train_X = torch.tensor(train_X, dtype=torch.float16)
    train_y = torch.tensor(train_y)
    val_X = torch.tensor(val_X, dtype=torch.float16)
    val_y = torch.tensor(val_y)
    test_X = torch.tensor(test_X, dtype=torch.float16)
    test_y = torch.tensor(test_y)

    train_loader = DataLoader(
        dataset=CNNDataset(train_X, train_y),
        batch_size=32
        )

    eval_loader = DataLoader(
        dataset=CNNDataset(val_X, val_y),
        batch_size=32
        )
    
    test_loader = DataLoader(
        dataset=CNNDataset(test_X, test_y),
        batch_size=32
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).half()
    loss_func = TLCLoss(cls_num_list=np.bincount(train_y)).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, eps=1e-2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = None

    num_class = len(torch.unique(train_y))
    weight_path = 'res/best.pt'
    current_precision, best_precision = 0, 0
    loss, acc = [], []
    print('Start training...')
    for epoch in range(epochs):
        train_loss = train_epoch(model, epoch, device, train_loader, loss_func, optimizer, scheduler)
        loss.append(train_loss)
        current_precision = valid_epoch(model, epoch, device, eval_loader, num_class)
        acc.append(current_precision)

        # if current_precision >= best_precision:
        #     best_precision = current_precision
        #     save_model_weight(model=model, filename=weight_path)
    
    print('\nCurrent best precision in val set is: %.4f' % (best_precision) + '%')
    # model.load_state_dict(torch.load(weight_path))
    acc = valid_epoch(model, epoch, device, test_loader, num_class)

    return model, acc

def train_epoch(model, epoch, device, train_loader, loss_func, optimizer, scheduler):
    model.train()
    total_loss = []
    for _, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            extra_info = {
                "num_expert"    : len(model.logits)   ,
                "logits"        : model.logits        ,
                'w'             : model.w
            }
            loss = loss_func(x=output,y=target,epoch=epoch,extra_info=extra_info)
            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
    batch_loss = 0.0
    if len(total_loss):
        batch_loss = sum(total_loss)/len(total_loss)
        print("loss =", batch_loss)
    if scheduler is not None:
        scheduler.step()
    return batch_loss


def valid_epoch(model, epoch, device, val_loader, num_class):
    model.eval()
    output = torch.empty(0, num_class, dtype=torch.float32, device=device)
    uncertainty = torch.empty(0, dtype=torch.float32, device=device)
    val_targets = []
    for _, (data, target) in enumerate(val_loader):
        data = data.to(device)
        with torch.no_grad():
            o = model(data)
            u = model.w[-1]
        output = torch.cat([output, o.detach()],dim=0)
        uncertainty = torch.cat([uncertainty, u.detach()], dim=0)
        val_targets.append(target)
    val_targets = torch.cat(val_targets, dim=0).to(device)
    print(f'================ Epoch: {epoch:03d} ================')
    ACC(output, val_targets, uncertainty, region_len=num_class/3)
