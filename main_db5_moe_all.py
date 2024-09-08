from main_db5_moe import main
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subject_range", default=[1, 10], nargs='+', help="an subjects range", type=int)
    parser.add_argument("--num_experts", default=3, help="display a cubic of a given number", type=int)
    parser.add_argument("--fusion", default=False, action="store_true", help="use BLUE fusion method")
    parser.add_argument("--uncertainty_type", default='DST', help="choose uncertainty type either DST or RSM", type=str)
    parser.add_argument("--device", default=0, help="GPU id", type=int)
    parser.add_argument("--reweight_epoch", default=30, help="epochs begin to reweight", type=int)
    parser.add_argument("--variable_cloud_size", default=False, action="store_true", help="wether use identical gaussian cloud size")
    parser.add_argument("--ucl_mul", default=1  / 2, help="cloud size multiplication", type=float)
    parser.add_argument("--adj_mul", default=1, help="adjust logits multiplication", type=float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    subjects_min = int(args.subject_range[0])
    subjects_max = int(args.subject_range[1])
    num_experts = int(args.num_experts)
    device = torch.device('cuda', args.device)
    reweight_epoch = int(args.reweight_epoch)
    fusion = args.fusion
    fusion_type = 'BLUE' if fusion else 'mean'
    uncertainty_type = args.uncertainty_type
    variable_cloud_size = args.variable_cloud_size
    ucl_mul = float(args.ucl_mul)
    adj_mul = float(args.adj_mul)

    cloud_size = 'variable' if variable_cloud_size else 'identical'
    model = f'{fusion_type}_{uncertainty_type}_{subjects_min}-{subjects_max}_{cloud_size}'

    root_dir = os.path.join('res', model)
    csv_dir = os.path.join(root_dir, 'csv')
    weight_dir = os.path.join(root_dir, 'weight')
    out_path = os.path.join(root_dir, 'output.txt')

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    acc_save_path = os.path.join(csv_dir, 'accuracy.csv')
    weight_path = os.path.join(weight_dir, 'best.pt')

    print(f'subject range: {subjects_min}-{subjects_max}')
    print(f'fusion: {fusion}')
    print(f'uncertainty_type: {uncertainty_type}')
    print(f'acc_save_path: {acc_save_path}')
    print(f'weight_path: {weight_path}')
    print(f'reweight_epoch: {reweight_epoch}')
    print(f'device: {device}')
    print(f'variable_cloud_size: {variable_cloud_size}')
    print(f'ucl_mul: {ucl_mul}')
    print(f'adj_mul: {adj_mul}')
 
    with open(acc_save_path, 'w') as f:
        f.write(',')
        for i in range(1, 1 + num_experts):
            f.write(str(i) + ',')
        for subject in range(subjects_min, subjects_max + 1):
            f.write('\n' + str(subject) + ',')
            for num_expert in range(1, 1 + num_experts):
                acc, region_acc, split_acc = main(subjects=subject, 
                                                  num_experts=num_expert, 
                                                  fusion=fusion,
                                                  weight_path=weight_path,
                                                  device=device,
                                                  uncertainty_type=uncertainty_type,
                                                  reweight_epoch=reweight_epoch,
                                                  variable_cloud_size=variable_cloud_size,
                                                  adj_mul=adj_mul,
                                                  ucl_mul=ucl_mul)
                f.write('%.2f' % (acc * 100) + '%,')
                f.flush()