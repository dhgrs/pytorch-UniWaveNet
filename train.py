import argparse
import pathlib

import tqdm
import torch
import tensorboardX

import params
from net import Encoder, UniWaveNet
from utils import DatasetFromFolder
from trainer import UniWaveNetTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='use cuda?')
parser.add_argument('--start_iteration', '-i', type=int, default=1,
                    help='Start iteraion setting for using resume')
parser.add_argument('--encoder_path', '-e', default=None,
                    help='Trained encoder path for using resum')
parser.add_argument('--wavenet_path', '-w', default=None,
                    help='Trained wavenet path for using resum')
parser.add_argument('--optimizer_path', '-o', default=None,
                    help='Optimizer state path for using resum')
args = parser.parse_args()

if args.use_cuda and not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --use_cuda')
device = torch.device('cuda' if args.use_cuda else 'cpu')

train_dataset = DatasetFromFolder(
    params.root, params.dataset_type, params.sr, params.length,
    params.frame_length, params.hop, params.n_mels, 'train',
    params.seed)
valid_dataset = DatasetFromFolder(
    params.root, params.dataset_type, params.sr, params.length,
    params.frame_length, params.hop, params.n_mels, 'valid',
    params.seed)
train_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=params.batch_size,
    shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=params.batch_size,
    shuffle=False)

encoder = Encoder(
    params.upscale_factors,
    params.n_wavenets * params.n_layers * params.n_loops, params.r,
    params.n_mels).to(device)
wavenet = UniWaveNet(
    params.n_wavenets, params.n_layers, params.n_loops, params.a, params.r,
    params.s).to(device)

optimizer = torch.optim.Adam(
    list(wavenet.parameters()) + list(encoder.parameters()), lr=params.lr)

train_writer = tensorboardX.SummaryWriter(
    str(pathlib.Path(params.output_dir, 'train')))
valid_writer = tensorboardX.SummaryWriter(
    str(pathlib.Path(params.output_dir, 'valid')))

trainer = UniWaveNetTrainer(
    train_data_loader, valid_data_loader, train_writer, valid_writer,
    params.valid_iteration, params.save_iteration, device, encoder, wavenet,
    optimizer, params.loss_weights, params.scale, params.loss_threshold, params.sr,
    params.output_dir, params.gradient_threshold)
trainer.load_trained_encoder(args.encoder_path)
trainer.load_trained_wavenet(args.wavenet_path)
trainer.load_optimizer_state(args.optimizer_path)

trainer.run(params.n_iteration)
