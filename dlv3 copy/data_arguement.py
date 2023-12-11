import librosa
import numpy as np


# 改速率（长度会变）
def time_stretch(x, rate): # rate：拉伸的尺寸， > 1 加快速度， < 1 放慢速度
    return librosa.effects.time_stretch(x, rate)


# 改音高（长度会变）
def resample(x, sr_o, rate):
    sr_g = int(sr_o * rate)
    return librosa.resample(x, sr_o, sr_g)


# 白噪声（叠加噪声，有可能超过允许范围）
def add_whitenoise(x, snr):                   # snr：生成的语音信噪比
    P_signal = np.sum(abs(x) ** 2) / len(x)    # 信号功率
    P_noise = P_signal / 10 ** (snr / 10.0)    # 噪声功率
    return x + np.random.randn(len(x)) * np.sqrt(P_noise)


# 基本音频数据增强：音高，速率，白噪声增强
def basic_voice_arguement(voice_array, sr, speed_rate, pitch_rate, snr):
    _voice_array = time_stretch(voice_array, speed_rate)
    _voice_array = resample(_voice_array, sr, pitch_rate)
    _voice_array = add_whitenoise(_voice_array, snr)
    return _voice_array


#  封装，在一定范围随机设定参数，默认参数以为合适的参数范围，不要轻易修改
def basic_voice_arguement_rand(voice_array, sr,
                               speed_rate_range=(0.9, 1.1), pitch_rate_range=(0.96, 1.04), snr_range=(18, 30)):
    speed_rate = (speed_rate_range[1] - speed_rate_range[0])*np.random.rand() + speed_rate_range[0]
    pitch_rate = (pitch_rate_range[1] - pitch_rate_range[0])*np.random.rand() + pitch_rate_range[0]
    snr = (snr_range[1] - snr_range[0])*np.random.rand() + snr_range[0]
    # print(speed_rate, pitch_rate, snr)
    _voice_array = basic_voice_arguement(voice_array, sr, speed_rate, pitch_rate, snr)
    return _voice_array

# 叠加指定环境噪声
# def add_env_noise(waveform, dB='81dB'):
#     wave_len = waveform.shape[0]
#     noise_wave = noises[dB]
#     noise_len = noise_wave.shape[0]
#     noise_start = np.random.randint(0,noise_len-wave_len-1)
#     noised_wave = noise_wave[noise_start:noise_start+wave_len] + waveform
#     return noised_wave
#
# noises_61dB,_ = librosa.load(r'D:\Necksense\数据中转\噪声测试\mic\device_noise_61dB_1.wav', sr=None)
# noises_81dB,_ = librosa.load(r'D:\Necksense\数据中转\噪声测试\s20\device_noise_81dB_2.wav', sr=None)
# noises = {'61dB': noises_61dB, '81dB': noises_81dB}