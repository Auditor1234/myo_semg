from main_db5_moe import main

if __name__ == '__main__':
    subjects_num = 10
    num_experts = 6
    fusion = True
    with open('res/csv/UCL_DST_6-10_accuracy.csv', 'w') as f:
        f.write(',')
        for i in range(1, 1 + num_experts):
            f.write(str(i) + ',')
        for subject in range(6, subjects_num + 1):
            f.write('\n' + str(subject) + ',')
            for num_expert in range(1, 1 + num_experts):
                acc, region_acc, split_acc = main(subjects=subject, num_experts=num_expert, fusion=fusion)
                f.write('%.2f' % (acc * 100) + '%,')
                f.flush()