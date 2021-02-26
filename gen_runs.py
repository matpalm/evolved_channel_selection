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


seed = 500
for dropout_channels in [False, True]:
    for learning_rate in [1e-2, 1e-3, 1e-4]:
        for batch_size in [32, 64, 128]:
            cmd = 'python3 train.py'
            cmd += ' --group dropout_channels'
            if dropout_channels:
                cmd += ' --dropout-channels'
            cmd += f" --seed {seed}"
            cmd += f" --learning-rate {learning_rate}"
            cmd += f" --batch-size {batch_size}"
            cmd += ' --epochs 20'
            cmd += ' --input-size 64'
            # print(f"ansible -i pod_inventory.ini cloud_tpu -a \"{cmd}\"")
            print(cmd)
            seed += 1
