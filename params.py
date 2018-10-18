import datetime

seed = None
root = '/media/hdd1/datasets/LJSpeech-1.1/'
dataset_type = 'LJSpeech'
sr = 22050
# root = '/media/hdd1/datasets/VCTK-Corpus/'
# dataset_type = 'VCTK'
# sr = 24000
length = 20480
batch_size = 1
frame_length = 1024
hop = 256
n_mels = 80
window = 'hann'

n_iteration = 1000000
lr = 1e-4
gradient_threshold = 10
loss_threshold = 100
valid_iteration = 10000
save_iteration = 10000

upscale_factors = [16, 16]
n_wavenets = 8
n_layers = 10
n_loops = 1
a = 128
r = 128
s = 128
scale = 'magnitude'
loss_weights = [0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.2, 0.3]

output_dir = \
    'results/{}'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
