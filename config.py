class linear_fc_config:
    def __init__(self, subject) -> None:
        if subject in [1, 4, 5]:
            self.ucl_noise_mul = 1 / 3
            self.cloud_size_mul = 5 / 3
        elif subject in [3, 9]:
            self.ucl_noise_mul = 1
            self.cloud_size_mul = 5 / 3
        elif subject in [10]:
            self.ucl_noise_mul = 1 / 4
            self.cloud_size_mul = 5 / 3
        elif subject in [2, 6]: # 6
            self.ucl_noise_mul = 1 / 2
            self.cloud_size_mul = 4 / 3
        elif subject in [7]:
            self.ucl_noise_mul = 1 / 4
            self.cloud_size_mul = 5 / 4
        elif subject in [8]:
            self.ucl_noise_mul = -1 / 4
            self.cloud_size_mul = 5 / 3

class nonlinear2_fc_config:
    def __init__(self, subject) -> None:
        self.adjust_mul = 1
        if subject in [2, 3, 10]:
            self.ucl_noise_mul = 1 / 4
            self.cloud_size_mul = 5 / 3
        elif subject in [1, 4]:
            self.ucl_noise_mul = 1 / 2
            self.cloud_size_mul = 4 / 3
        elif subject in [6, 8]:
            self.ucl_noise_mul = -1 / 2
            self.cloud_size_mul = 4 / 3
        elif subject in [5]:
            self.ucl_noise_mul = 1 / 5
            self.cloud_size_mul = 1
            self.adjust_mul = 0
        elif subject in [7]:
            self.ucl_noise_mul = 1 / 2
            self.cloud_size_mul = 5 / 4
            self.adjust_mul = 1 / 5
        elif subject in [9]:
            self.ucl_noise_mul =  -1 / 4
            self.cloud_size_mul = 5 / 3
            self.adjust_mul = 3 / 2

class nonlinear3_fc_config:
    def __init__(self, subject) -> None:
        self.cloud_size_mul = 1
        if subject in [7]:
            self.ucl_mul =  1 / 2
            self.adj_mul = 1
        elif subject in [3]:
            self.ucl_mul =  1
            self.adj_mul = 1
        elif subject in [9]:
            self.ucl_mul =  1 / 4
            self.adj_mul = 1
        elif subject in [4]:
            self.ucl_mul =  1 / 3
            self.adj_mul = 1 / 2
        elif subject in [2]:
            self.ucl_mul =  1 / 4
            self.adj_mul = 1 / 2
        elif subject in [6]:
            self.ucl_mul =  1 / 6
            self.adj_mul = 1 / 2
        elif subject in [8]:
            self.ucl_mul =  1 / 5
            self.adj_mul = 1.2
        elif subject in [10]:
            self.ucl_mul =  1
            self.adj_mul = 1.1
        elif subject in [1]:
            self.ucl_mul =  0.2
            self.adj_mul = 0.5
        elif subject in [5]:
            self.ucl_mul =  1
            self.adj_mul = 1.1

if __name__ == '__main__':
    param_config = linear_fc_config('subject8')
    print(param_config.cloud_size_mul)