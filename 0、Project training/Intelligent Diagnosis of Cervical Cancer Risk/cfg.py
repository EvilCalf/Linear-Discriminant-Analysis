batch_size = 8
patch_size = [800, 800]
epochs = 500
start_lr = 0.001
lr_power = 0.9
weight_decay = 0.0001
num_worker = 4

alpha = 0.25
gamma = 0.75


backend = "retinanet"

train_path = "data/train/"
test_path = "data/test/"

sample_path = "../samples/"
visual_sample_path = ""  # change to validation sample path (including .npz files)
checkpoint_path = "../checkpoint/"
log_path = "../log/"
result_path = "../result/"
