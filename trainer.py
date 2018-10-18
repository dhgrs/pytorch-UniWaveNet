import pathlib

import torch
import tqdm

from utils import Trainer


class UniWaveNetTrainer(Trainer):
    def __init__(
            self, train_data_loader, valid_data_loader, train_writer,
            valid_writer, valid_iteration, save_iteration, device, encoder,
            wavenet, optimizer, loss_weights, scale, loss_threshold, sr,
            output_dir, gradient_threshold):
        super(UniWaveNetTrainer, self).__init__(
            train_data_loader, valid_data_loader, train_writer, valid_writer,
            valid_iteration, save_iteration, device)
        self.encoder = encoder
        self.wavenet = wavenet
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scale = scale
        self.loss_threshold = loss_threshold
        self.sr = sr
        self.output_dir = output_dir
        self.gradient_threshold = gradient_threshold

    def _train_one_iteration(self, iteration):
        batch = next(iter(self.train_data_loader))
        t, spectrogram = batch[0].to(self.device), batch[1].to(self.device)
        conditions = self.encoder(spectrogram)
        xs = self.wavenet(conditions, return_all=True)

        magnitude_loss = 0
        power_loss = 0
        log_loss = 0
        for x in xs:
            magnitude_loss += calc_spectrogram_loss(
                x, t, 'magnitude', self.loss_weights, self.loss_threshold)
            power_loss += calc_spectrogram_loss(
                x, t, 'power', self.loss_weights, self.loss_threshold)
            log_loss += calc_spectrogram_loss(
                x, t, 'log', self.loss_weights, self.loss_threshold)

        if self.scale == 'magnitude':
            loss = magnitude_loss
        elif self.scale == 'power':
            loss = power_loss
        else:
            loss = log_loss

        self.train_writer.add_audio(
            'audio/generated', torch.clamp(xs[-1][0], -1, 1), iteration,
            sample_rate=self.sr)
        self.train_writer.add_audio(
            'audio/grandtruth', torch.clamp(t[0], -1, 1), iteration,
            sample_rate=self.sr)

        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_threshold is not None:
            torch.nn.utils.clip_grad_norm_(
                self.wavenet.parameters(), self.gradient_threshold)
        self.optimizer.step()
        self.train_writer.add_scalar(
            'magnitude_loss', magnitude_loss.item(), iteration)
        self.train_writer.add_scalar(
            'power_loss', power_loss.item(), iteration)
        self.train_writer.add_scalar(
            'log_loss', log_loss.item(), iteration)
        self.train_writer.add_scalar(
            'loss', loss.item(), iteration)

    def _valid(self, iteration):
        avg_magnitude_loss = 0
        avg_power_loss = 0
        avg_log_loss = 0
        avg_loss = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(self.valid_data_loader):
                t, spectrogram = \
                    batch[0].to(self.device), batch[1].to(self.device)
                conditions = self.encoder(spectrogram)
                xs = self.wavenet(conditions, return_all=True)

                magnitude_loss = 0
                power_loss = 0
                log_loss = 0
                for x in xs:
                    magnitude_loss += calc_spectrogram_loss(
                        x, t, 'magnitude', self.loss_weights,
                        self.loss_threshold)
                    power_loss += calc_spectrogram_loss(
                        x, t, 'power', self.loss_weights,
                        self.loss_threshold)
                    log_loss += calc_spectrogram_loss(
                        x, t, 'log', self.loss_weights,
                        self.loss_threshold)

                if self.scale == 'magnitude':
                    loss = magnitude_loss
                elif self.scale == 'power':
                    loss = power_loss
                else:
                    loss = log_loss

                avg_magnitude_loss += magnitude_loss.item()
                avg_power_loss += power_loss.item()
                avg_log_loss += log_loss.item()
                avg_loss += loss.item()
                self.valid_writer.add_audio(
                    'audio/generated', torch.clamp(xs[-1][0], -1, 1),
                    iteration, sample_rate=self.sr)
                self.valid_writer.add_audio(
                    'audio/grandtruth', torch.clamp(t[0], -1, 1),
                    iteration, sample_rate=self.sr)
        self.valid_writer.add_scalar(
            'magnitude_loss', avg_magnitude_loss / len(self.valid_data_loader),
            iteration)
        self.valid_writer.add_scalar(
            'power_loss', avg_power_loss / len(self.valid_data_loader),
            iteration)
        self.valid_writer.add_scalar(
            'log_loss', avg_log_loss / len(self.valid_data_loader), iteration)
        self.valid_writer.add_scalar(
            'loss', avg_loss / len(self.valid_data_loader), iteration)

    def _save_checkpoints(self, iteration):
        def _save_checkpoint(model, model_name):
            model_out_path = '{}_iteration_{}.pth'.format(
                model_name, iteration)
            torch.save(model.state_dict(), model_out_path)
            print('Checkpoint is saved to {}'.format(model_out_path))
        _save_checkpoint(
            self.wavenet, str(pathlib.Path(self.output_dir, 'wavenet')))
        _save_checkpoint(
            self.encoder, str(pathlib.Path(self.output_dir, 'encoder')))
        _save_checkpoint(
            self.optimizer, str(pathlib.Path(self.output_dir, 'optimizer')))

    def load_trained_encoder(self, model_path):
        if model_path is not None:
            self.encoder.load_state_dict(torch.load(model_path))

    def load_trained_wavenet(self, model_path):
        if model_path is not None:
            self.wavenet.load_state_dict(torch.load(model_path))

    def load_optimizer_state(self, optimizer_path):
        if optimizer_path is not None:
            self.optimizer.load_state_dict(torch.load(optimizer_path))


def calc_spectrograms(signal, scale):
    signal = torch.squeeze(signal, dim=1)
    spectrograms = []
    for n_fft in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        hop_length = n_fft // 4
        win_length = n_fft // 2
        if torch.cuda.is_available():
            window = torch.hann_window(win_length).cuda()
        else:
            window = torch.hann_window(win_length)
        complex_spectrogram = torch.stft(
            signal, n_fft, hop_length, win_length, window)
        # [minibatch, frame, frequency, real/imaginary]
        power_spectrogram = (
            complex_spectrogram[:, :, :, 0] * complex_spectrogram[:, :, :, 0] +
            complex_spectrogram[:, :, :, 1] * complex_spectrogram[:, :, :, 1])
        if scale == 'power':
            spectrograms.append(power_spectrogram)
        elif scale == 'magnitude':
            spectrograms.append(torch.sqrt(power_spectrogram + 1e-10))
        elif scale == 'log':
            spectrograms.append(torch.log(power_spectrogram + 1e-10))
        else:
            print('error')
    return spectrograms


def calc_spectrogram_loss(x, t, scale, weights, loss_threshold=100):
    x_specs = calc_spectrograms(x, scale)
    t_specs = calc_spectrograms(t, scale)
    loss = 0
    for x_spec, t_spec, weight in zip(x_specs, t_specs, weights):
        loss += weight * torch.mean(
            torch.clamp(torch.abs(x_spec - t_spec), max=loss_threshold))
    return loss
