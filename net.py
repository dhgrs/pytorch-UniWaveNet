import torch


class Encoder(torch.nn.Module):
    def __init__(self, upscaling_factors, n_layers, r, n_mels):
        super(Encoder, self).__init__()
        deconv_list = []
        for factor in upscaling_factors:
            deconv_list.append(torch.nn.ConvTranspose2d(
                1, 1, (3, 2 * factor), (1, factor), (1, factor // 2)))
        projection_list = []
        for layer in range(n_layers):
            projection_list.append(torch.nn.Conv1d(n_mels, 2 * r, 1))
        self.deconv_list = torch.nn.ModuleList(deconv_list)
        self.projection_list = torch.nn.ModuleList(projection_list)

    def forward(self, x):
        for deconv in self.deconv_list:
            x = torch.nn.functional.leaky_relu(deconv(x), 0.4)
            if deconv.stride[1] % 2 != 0:
                x = x[:, :, :, :-1]
        x = torch.squeeze(x, dim=1)
        projected = []
        for projection in self.projection_list:
            projected.append(projection(x))
        return projected


class WaveNet(torch.nn.Module):
    def __init__(self, n_layers, n_loops, a, r, s):
        super(WaveNet, self).__init__()
        self.conv_in = torch.nn.Conv1d(1, r, 3, 1, 1)
        dilation_list = []
        skip_list = []
        residual_list = []
        for loop in range(n_loops):
            for layer in range(n_layers):
                dilation_list.append(torch.nn.Conv1d(
                    r, 2 * r, 3, 1,
                    padding=3 ** layer, dilation=3 ** layer))
                skip_list.append(torch.nn.Conv1d(
                    r, s, 1))
                residual_list.append(torch.nn.Conv1d(
                    r, r, 1))
        self.dilation_list = torch.nn.ModuleList(dilation_list)
        self.skip_list = torch.nn.ModuleList(skip_list)
        self.residual_list = torch.nn.ModuleList(residual_list)
        self.conv1x1 = torch.nn.Conv1d(s, a, 1)
        self.conv_out = torch.nn.Conv1d(a, 1, 1)

    def forward(self, x, conditions):
        x = torch.tanh(self.conv_in(x))
        if torch.isnan(self.conv_in.weight).any():
            x.data[...] = -1
        skip_connection = 0
        for i, (dilation, skip, residual, condition) in enumerate(zip(
                self.dilation_list, self.skip_list, self.residual_list,
                conditions)):
            z = dilation(x)
            z += condition
            z_tanh, z_sigmoid = torch.chunk(z, 2, dim=1)
            z = torch.tanh(z_tanh) * torch.sigmoid(z_sigmoid)
            skip_connection += skip(z)
            x = x + residual(z)
        x = torch.nn.functional.relu(skip_connection)
        x = torch.nn.functional.relu(self.conv1x1(x))
        y = self.conv_out(x)
        return y


class UniWaveNet(torch.nn.Module):
    def __init__(self, n_wavenets, *args, **kwargs):
        super(UniWaveNet, self).__init__()
        self.wavenet_list = torch.nn.ModuleList(
            [WaveNet(*args, **kwargs) for i in range(n_wavenets)])

    def forward(self, conditions, return_all=False):
        batchsize = conditions[0].size(0)
        wave_length = conditions[0].size(2)
        x = self._generate_random(
            (batchsize, 1, wave_length)).to(conditions[0])
        generated = []
        layer_per_wavenet = len(conditions) // len(self.wavenet_list)
        for i, wavenet in enumerate(self.wavenet_list[1:]):
            x = wavenet(
                x,
                conditions[i * layer_per_wavenet:(i + 1) * layer_per_wavenet])
            generated.append(x)
        if return_all:
            return generated
        else:
            return x

    def _generate_random(self, shape):
        base_distribution = torch.distributions.Uniform(0, 1)
        transforms = [
            torch.distributions.transforms.SigmoidTransform().inv,
            torch.distributions.transforms.AffineTransform(loc=0, scale=0.05)]
        logistic = torch.distributions.TransformedDistribution(
            base_distribution, transforms)
        return logistic.sample(shape)
