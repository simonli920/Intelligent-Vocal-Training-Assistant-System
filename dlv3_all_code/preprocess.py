import numpy as np
import librosa
import scipy.signal as signnal
from sff import SFF
from data_arguement import basic_voice_arguement_rand

def window_cut(voice_array, sample_rate, lens_win=200, step = 100, arguement=0):

    voice_array = normalization(voice_array)
    i = 0
    lens_voice = len(voice_array)
    lens_win = int(lens_win*sample_rate/1000)
    step = int(step*sample_rate/1000)
    while 1:
        if i*step+lens_win > lens_voice:
            if lens_voice < lens_win:
                return []
                # if rand_pad == False:
                #     voice_array = np.pad(voice_array, (0, lens_win-lens_voice), 'constant', constant_values=(0, 0))
                # elif rand_pad == True:
                #     voice_array = rand_pad(voice_array, (0, lens_win-lens_voice), miu=0, sigma=0.001)
                # return [voice_array]
            else:
                break

        yield voice_array[i*step:i*step+lens_win]
        i += 1


def normalization(voice):
    _max = voice.max()
    _min = voice.min()
    scale = max(abs(_max), abs(_min))
    return voice/scale


def rand_pad(np_array, pad_num = (0,0), miu=0, sigma = 0.001):
    left = np.random.normal(miu, sigma, pad_num[0])
    right = np.random.normal(miu, sigma, pad_num[1])
    new_array = np.append(left, np_array)
    new_array = np.append(new_array, right)
    return new_array


# 注意lens必须是偶数整数，单位为ms
# 从音频序列正中间切出固定长度的一部分，参数为：音频序列，采样率，切出片段长度（单位ms），
# 可选填充类型：零填充'zero',小噪声填充'rand',舍去'none'
def middle_cut(voice_array, sample_rate, lens=200, pad_type='zero'):
    i = 0
    lens_voice = len(voice_array)
    lens_segment = int(lens*sample_rate/1000)
    if lens_segment > lens_voice:
        if pad_type == 'none':
            return None
        else:
            num_padding = int((lens_segment - lens_voice) / 2)
            if pad_type == 'zero':
                segment = np.pad(voice_array, (num_padding, lens_segment-num_padding-lens_voice), 'constant', constant_values=(0, 0))
            elif pad_type == 'rand':
                segment = rand_pad(voice_array, (num_padding, lens_segment-num_padding-lens_voice), miu=0, sigma=0.001)
            return segment

    else:
        idx_middle = int(lens_voice/2)
        num_range = int(lens_segment/2)
        segment = voice_array[idx_middle-num_range:idx_middle+num_range]
        return segment

# 取音频中间位置的部分作为样本，可选择增强次数
def middle_samples(voice_array, sample_rate, lens=200, pad_type='zero', n_arguement=0):
    raw_segment = middle_cut(voice_array, sample_rate, lens=lens, pad_type=pad_type)
    raw_segment = normalization(raw_segment)
    argue_segments = [raw_segment]
    for i in range(n_arguement):
        _argue_voice_array = basic_voice_arguement_rand(voice_array, sample_rate)
        _argue_segment = middle_cut(_argue_voice_array, sample_rate, lens=lens, pad_type=pad_type)
        _argue_segment = normalization(_argue_segment)
        argue_segments.append(_argue_segment)
    return argue_segments

# 临时切分，只在此函数修改切分方法
def temp_cut(voice_array):
    # 从音频序列正中间切出一条样本，参数为：音频序列，采样率，切出片段长度（单位ms），填充类型：'zero','rand','none'
    voice_array = normalization(voice_array)

    return [voice_array]


# 下面是谱变换函数
def stft_specgram(waveform, sample_rate, n_fft=256, step=128):
    _, _, specgram = signnal.stft(waveform, sample_rate, nperseg=n_fft, noverlap=step)
    specgram = np.abs(specgram)
    specgram = 10*np.log10(np.abs(specgram)*np.abs(specgram))
    # 保留人声频率范围附近的部分
    max_freq = int((2 * 8000 / sample_rate) * specgram.shape[0])
    specgram = specgram[:max_freq, :]
    print(specgram.shape)
    return specgram


def mel_specgram(waveform, sample_rate, n_mels=256, hop_length=128):
    specgram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels, hop_length=hop_length)
    specgram = librosa.power_to_db(specgram)
    return specgram

def mfccs(waveform, sample_rate, n_mfcc=40, hop_length=256):
    _mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length)
    # print(_mfccs.shape)
    # return _mfccs.flatten()
    return _mfccs

def sff_specfram(waveform, sample_rate, n_f=256, n_t=256):
    return SFF(waveform, sample_rate, n_f=n_f, n_t=n_t)

def lpc_specgram(waveform, win_size = 256, step = 128):
    win_start = 0
    win_end = win_size
    while win_end <= waveform.size:
#         print(win_start,win_end)
        segment = waveform[win_start:win_end]
#         print(segment[:10].dtype)
        lpc_cof = librosa.lpc(segment, 128)
        if win_start == 0:
            specgram = lpc_cof
        else:
            specgram = np.row_stack((specgram,lpc_cof))
        win_start += step
        win_end += step
    return specgram

specgrams = {}
specgrams['stft'] = stft_specgram
specgrams['mel'] = mel_specgram
specgrams['mfccs'] = mfccs
specgrams['sff'] = sff_specfram
specgrams['lpc'] = lpc_specgram



