# pytorch-UniWaveNet

A PyTorch implementetion of Uni-WaveNet( https://zhuanlan.zhihu.com/p/44702855 ). Uni-WaveNet is derived from Parallel WaveNet, it can be trained without Teacher WaveNet.

# Generated samples

I uploaded generated samples to SoundCloud.
https://soundcloud.com/dhgrs/sets/uni-wavenet

And uploaded pretrained model for LJSpeech to Google Drive, same setting as [NVIDIA's Tacotron2 implementaiton](https://github.com/NVIDIA/tacotron2).
https://drive.google.com/drive/folders/1BqzltOT9u3358nQgPolRA511J7cGJGlI?usp=sharing

# Requirements

- Python3
- PyTorch(>0.4.1)
- tensorboardX
- numpy
- libsosa
- tqdm

**CAUTION**: The interface of PyTorch's STFT API has changed at ver. 0.4.1. So you have to use 0.4.1 or later.

# Usage
1. Download dataset

This implementation can train with LJSpeech, English single speaker corpus or VCTK-Corpus, English multi speaker corpus. And you can download them very easily via [my repository](https://github.com/dhgrs/download_dataset).

2. Set parameters

Hyperparameters are in `params.py`. You have to change `root` to the directory you download the dataset. If you can not understand some parameters, please open an issue.

3. Run training

```
# without GPU
python3 train.py

# with GPU
python3 train.py --use_cuda
```
If you want to restart training with snapshot, use options like below.
```
python3 train.py --use_cuda -e path/to/encoder.pth -w path/to/wavenet.pth -o path/to/optimizer.pth -i iteration_to_start_at
```
You can use TensorBoard to visualize training. Also can listen the generated samples during training.

4. Generate

```
python3 generate.py -e path/to/encoder.pth -w path/to/wavenet.pth -i path/to/input.wav -l length_to_generate(sec)
```

**CAUTION**: There is a bug on PyTorch and my implementation. Now you can generate samples on ONLY GPU. You will get an error if you try to generate on CPU. Related issue which I opened is [here](https://github.com/pytorch/pytorch/issues/12484).

