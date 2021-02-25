#!/usr/bin/env python3


# def extra_flags_for(opt):
#     if opt == 'lamb':
#         for weight_decay in [0.0, 1e-1, 1e-2, 1e-3]:
#             yield f" --weight-decay {weight_decay}"
#     elif opt == 'sgd':
#         for momentum in [0, 0.9, 0.99]:
#             yield f" --momentum {momentum}"
#             yield f" --momentum {momentum} --nesterov"
#     else:  # adam
#         yield ""  # has nothing extra


seed = 456
for learning_rate in [1e-2, 1e-3, 1e-4]:
    for input_size in [64, 32, 16, 8]:
        cmd = 'python3 train.py --group input_size_sweep'
        cmd += f" --seed {seed}"
        cmd += f" --learning-rate {learning_rate}"
        cmd += ' --batch-size 64'
        cmd += ' --epochs 10'
        cmd += f" --input-size {input_size}"
        # print(f"ansible -i pod_inventory.ini cloud_tpu -a \"{cmd}\"")
        print(cmd)
        seed += 1
