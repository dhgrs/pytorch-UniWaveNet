import argparse

import torch
import numpy
import librosa

import params
from net import Encoder, UniWaveNet
from utils import DatasetFromFolder

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', help='use cuda?')
parser.add_argument('--wavenet_model', '-w',
                    help='Trained WaveNet model')
parser.add_argument('--encoder_model', '-e',
                    help='Trained Encoder model')
parser.add_argument('--length', '-l', type=int, default=5,
                    help='Length in seconds to generate')
parser.add_argument('--input', '-i', help='Input file name')
parser.add_argument('--output', '-o', default='result.wav',
                    help='Output file name')

args = parser.parse_args()

if args.use_cuda and not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --use_cuda')
device = torch.device('cuda' if args.use_cuda else 'cpu')

dataset = DatasetFromFolder(
    args.input, 'file', params.sr, params.sr * args.length,
    params.frame_length, params.hop, params.n_mels, 'valid',
    None)

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

encoder = Encoder(
    params.upscale_factors,
    params.n_wavenets * params.n_layers * params.n_loops, params.r,
    params.n_mels)
wavenet = UniWaveNet(
    params.n_wavenets, params.n_layers, params.n_loops, params.a, params.r,
    params.s)
encoder = encoder.to(device)
wavenet = wavenet.to(device)

encoder.load_state_dict(torch.load(args.encoder_model))
wavenet.load_state_dict(torch.load(args.wavenet_model))

_, spectrogram = next(iter(data_loader))
spectrogram = spectrogram.to(device)
with torch.no_grad():
    conditions = encoder(spectrogram)
    x = wavenet(conditions)
    x = x.cpu()
librosa.output.write_wav(
    args.output, x[0, 0].numpy().astype(numpy.float32), sr=params.sr)
