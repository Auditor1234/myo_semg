from main_db5_moe import main
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subject_range", default=[1, 10], nargs='+', help="an subjects range", type=int)
    parser.add_argument("--num_experts", default=6, help="display a cubic of a given number", type=int)
    parser.add_argument("--fusion", default=True, action="store_true", help="use BLUE fusion method")
    parser.add_argument("--uncertainty_type", default='DST', help="choose uncertainty type either DST or RSM", type=str)
    parser.add_argument("--device", default=0, help="GPU id", type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    subjects_min = int(args.subject_range[0])
    subjects_max = int(args.subject_range[1])
    num_experts = int(args.num_experts)
    device = torch.device('cuda', args.device)
    fusion = args.fusion
    fusion_type = 'BLUE' if fusion else 'mean'
    uncertaint_type = args.uncertainty_type
    model = f'UCL_{fusion_type}_{uncertaint_type}_{subjects_min}-{subjects_max}'

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
    print(f'uncertainty_type: {uncertaint_type}')
    print(f'acc_save_path: {acc_save_path}')
    print(f'weight_path: {weight_path}')
 
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
                                                  reweight_epoch=20)
                f.write('%.2f' % (acc * 100) + '%,')
                f.flush()