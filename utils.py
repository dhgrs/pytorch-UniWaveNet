import pathlib
import random

import tqdm
import librosa
import numpy
import torch
import torch.utils.data


class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(
            self, root_or_path, dataset_type, sr, length, n_fft, hop_length,
            n_mels, train_or_valid, seed):
        super(DatasetFromFolder, self).__init__()
        paths = self.get_sorted_paths(root_or_path, dataset_type)
        if seed is not None:
            random.seed(seed)
            paths = random.sample(paths, len(paths))
        n_train = int(len(paths) * 0.9)
        if train_or_valid == 'train':
            self.paths = paths[:n_train]
        elif train_or_valid == 'valid':
            self.paths = paths[n_train:]
        self.dataset_type = dataset_type
        self.sr = sr
        self.length = length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __getitem__(self, index):
        path = self.paths[index]
        x, _ = librosa.load(path, sr=self.sr)
        x /= numpy.abs(x).max()
        x, _ = librosa.effects.trim(x, top_db=20)
        if len(x) <= self.length:
            pad = self.length - len(x)
            x = numpy.concatenate((x, numpy.zeros(pad)))
        else:
            start_idx = random.randint(0, len(x) - self.length - 1)
            end_idx = start_idx + self.length
            x = x[start_idx:end_idx]

        mel_spectrogram = librosa.feature.melspectrogram(
            x, self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels)
        if self.length is not None:
            mel_spectrogram = \
                mel_spectrogram[:, :self.length // self.hop_length]
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=numpy.max)
        mel_spectrogram += 80
        mel_spectrogram /= 80

        x = numpy.expand_dims(x, axis=0)
        mel_spectrogram = numpy.expand_dims(mel_spectrogram, axis=0)
        return x.astype(numpy.float32), mel_spectrogram.astype(numpy.float32)

    def __len__(self):
        return len(self.paths)

    def get_sorted_paths(self, root_or_path, dataset_type):
        if dataset_type == 'VCTK':
            paths = sorted([
                path for path
                in pathlib.Path(root_or_path).glob('wav48/*/*.wav')])
        elif dataset_type == 'LJSpeech':
            paths = sorted([
                path for path
                in pathlib.Path(root_or_path).glob('wavs/*.wav')])
        elif dataset_type == 'file':
            paths = [root_or_path]
        return paths


class Trainer(object):
    def __init__(
            self, train_data_loader, valid_data_loader, train_writer,
            valid_writer, valid_iteration, save_iteration, device):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.train_writer = train_writer
        self.valid_writer = valid_writer
        self.valid_iteration = valid_iteration
        self.save_iteration = save_iteration
        self.device = device

    def _train_one_iteration(self, iteration):
        pass

    def _valid(self, iteration):
        pass

    def _save_checkpoints(self, iteration):
        pass

    def run(self, total_iteration, start_iteration=1):
        progress_bar = tqdm.tqdm(
            range(start_iteration, start_iteration + total_iteration))
        for iteration in progress_bar:
            self._train_one_iteration(iteration)

            if iteration % self.save_iteration == 0:
                self._save_checkpoints(iteration)

            if iteration % self.valid_iteration == 0:
                self._valid(iteration)
