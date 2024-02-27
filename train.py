import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm
from window_dataloader import CNNDataset
from common_utils import save_model_weight
from loss import combine_loss



def train(model, epochs, train_X, train_y, val_X, val_y, test_X, test_y, evidential=False):
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
    if evidential:
        loss_func = combine_loss
    else:
        loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, eps=1e-2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = None

    weight_path = 'res/best.pt'
    current_precision, best_precision = 0, 0
    loss, acc = [], []
    print('Start training...')
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, epoch, epochs, device, train_loader, loss_func, optimizer, scheduler)
        loss.append(train_loss)
        current_precision = validate(model, device, eval_loader, loss_func)
        acc.append(current_precision)

        if current_precision >= best_precision:
            best_precision = current_precision
            save_model_weight(model=model, filename=weight_path)
    
    print('\nCurrent best precision in val set is: %.4f' % (best_precision) + '%')
    model.load_state_dict(torch.load(weight_path))
    acc = evaluate(model, device, test_loader, loss_func)

    return model, acc


def train_one_epoch(model, epoch, epochs, device, train_loader, loss_func, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc='Train')
    for _, (x_data, y_label) in enumerate(loop): # shape(B, 12, 200, 1)

        x_data = x_data.to(device)
        
        predict = model(x_data)

        labels = y_label.type(torch.long).to(device)
        loss = loss_func(predict, labels)
        total_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loop.set_postfix(loss = loss.item())

    if scheduler != None:
        scheduler.step()
    print("[%d/%d] epoch's average loss = %.6f" % (epoch + 1, epochs, total_loss / len(train_loader)))
    return total_loss.item() / len(train_loader)


def validate(model, device, val_loader, loss_func):
    model.eval()
    total_loss, correct_nums, total_nums = 0.0, 0, 0
    
    print("Validating...")
    loop = tqdm(val_loader, desc='Validation')
    for i, (x_data, y_label) in enumerate(loop): # shape(4, *, 12)
        
        predict = model(x_data.to(device))
        
        labels = y_label.type(torch.long).to(device)
        # loss = loss_func(predict, labels)
        # total_loss += loss

        predict_idx = predict.argmax(dim=-1)
        correct_nums += torch.sum(predict_idx == labels)
        total_nums += len(predict_idx)

        loop.set_description(f'Validation [{i + 1}/{len(val_loader)}]')
        # loop.set_postfix(loss = loss.item())
    
    acc = round(100.0 * correct_nums.item() / total_nums, 6)
    print("Validation total loss: %.6f," % (total_loss), "Accuracy: %.4f" % (acc) + '%\n')
    return acc



def evaluate(model, device, eval_loader, loss_func):
    model.eval()
    total_loss, correct_nums, total_nums = 0.0, 0, 0
    
    print("Evaluating...")
    loop = tqdm(eval_loader, desc='Evaluation')
    for i, (x_data, y_label) in enumerate(loop): # shape(4, *, 12)
        
        predict = model(x_data.to(device))
        
        labels = y_label.type(torch.long).to(device)
        # loss = loss_func(predict, labels)
        # total_loss += loss

        predict_idx = predict.argmax(dim=-1)
        correct_nums += torch.sum(predict_idx == labels)
        total_nums += len(predict_idx)

        loop.set_description(f'Evaluating [{i + 1}/{len(eval_loader)}]')
        # loop.set_postfix(loss = loss.item())
    
    acc = round(100.0 * correct_nums.item() / total_nums, 6)
    print("Evaluation total loss: %.6f," % (total_loss), "Accuracy: %.4f" % (acc) + '%')
    return acc

