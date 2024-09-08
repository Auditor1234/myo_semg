# resemble data
import os
from glob import glob
from functools import cmp_to_key
import time
import os
import subprocess
import threading


def cmp(file1, file2):
    if len(file1) > len(file2):
        return 1
    elif len(file1) < len(file2):
        return -1
    elif file1 > file2:
        return 1
    else:
        return -1

def parse_acc(expert_acc_sum, line):
    accuracies = line.strip().split(',')
    for i in range(len(accuracies)):
        if '%' not in accuracies[i]:
            continue
        acc = eval(accuracies[i].rstrip("%"))
        expert_acc_sum[i - 1] += acc

def save_results(results_root='res', results_name='variable_60_ucl_mul_1_adj_0.5.csv', mode='/*BLUE*'):
    results_file = os.path.join(results_root, results_name)
    trained_dirs = sorted(glob(results_root + mode), key=cmp_to_key(cmp))
    expert_acc_sum = [0, 0, 0]
    subject_num = 0
    with open(results_file, 'w') as result:
        for i in range(len(trained_dirs)):
            file = os.path.join(trained_dirs[i], 'csv/accuracy.csv')
            with open(file, 'r') as source:
                lines = source.readlines()
                subject_num += len(lines) - 1
                for j in range(1, len(lines)):
                    result.write(lines[j])
                    parse_acc(expert_acc_sum, lines[j])
            result.write('\n')
        average_acc = ''
        for i in range(len(expert_acc_sum)):
            average_acc += '%.2f' % (expert_acc_sum[i] / subject_num) + '%,'
        result.write(',' + average_acc)


def run_db5(cmd):
    start_time = time.time()
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    end_time = time.time()
    print(cmd, 'process time: %.2f' % (end_time - start_time), 's')

def multi_thread_fun(fusion='--fusion', variable_cloud_size='--variable_cloud_size', u_type='DST', adj_mul=1, ucl_mul=1):
    cmds = [
        f'python main_db5_moe_all.py --subject_range 1 4 {fusion} {variable_cloud_size} --uncertainty_type={u_type} --adj_mul={adj_mul} --ucl_mul={ucl_mul}',
        f'python main_db5_moe_all.py --subject_range 5 7 {fusion} {variable_cloud_size} --uncertainty_type={u_type} --adj_mul={adj_mul} --ucl_mul={ucl_mul}',
        f'python main_db5_moe_all.py --subject_range 8 10 {fusion} {variable_cloud_size} --uncertainty_type={u_type} --adj_mul={adj_mul} --ucl_mul={ucl_mul}',
    ]

    # 并行
    threads = []
    for cmd in cmds:
        th = threading.Thread(target=run_db5, args=(cmd,))
        th.start()
        threads.append(th)
    # 等待线程运行完毕
    for th in threads:
        th.join()
    