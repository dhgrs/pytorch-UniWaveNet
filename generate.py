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
parser.add_argument('--input', '-i', help='Input file name')
parser.add_argument('--output', '-o', default='result.wav',
                    help='Output file name')

args = parser.parse_args()

if args.use_cuda and not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')
device = torch.device('cuda' if args.use_cuda else 'cpu')

dataset = DatasetFromFolder(
    args.input, 'file', params.sr, params.length,
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

print(torch.load(args.wavenet_model)['wavenet_list.0.conv_in.weight'][0][0][0])
print(torch.load(args.wavenet_model)['wavenet_list.0.conv1x1.weight'][0][0][0])
print(torch.load(args.wavenet_model)['wavenet_list.0.conv_out.weight'][0][0][0])

print(torch.load(args.wavenet_model)['wavenet_list.1.conv_in.weight'][0][0][0])
print(torch.load(args.wavenet_model)['wavenet_list.1.conv1x1.weight'][0][0][0])
print(torch.load(args.wavenet_model)['wavenet_list.1.conv_out.weight'][0][0][0])

print(torch.load(args.wavenet_model)['wavenet_list.2.conv_in.weight'][0][0][0])
print(torch.load(args.wavenet_model)['wavenet_list.2.conv1x1.weight'][0][0][0])
print(torch.load(args.wavenet_model)['wavenet_list.2.conv_out.weight'][0][0][0])

print(torch.load(args.wavenet_model)['wavenet_list.3.conv_in.weight'][0][0][0])
print(torch.load(args.wavenet_model)['wavenet_list.3.conv1x1.weight'][0][0][0])
print(torch.load(args.wavenet_model)['wavenet_list.3.conv_out.weight'][0][0][0])
# print(encoder.deconv_list[0].weight)
# print(wavenet.wavenet_list[0].dilation_list[0].weight)

_, spectrogram = next(iter(data_loader))
spectrogram = spectrogram.to(device)
with torch.no_grad():
    conditions = encoder(spectrogram)
    x = wavenet(conditions)
librosa.output.write_wav(
    args.output, x[0, 0].numpy().astype(numpy.float32), sr=params.sr)
