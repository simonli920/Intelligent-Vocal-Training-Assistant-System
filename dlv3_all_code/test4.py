from plottool import plot_specgram
from preprocess import middle_cut,stft_specgram,mel_specgram
import librosa
import os


test_dir_path = r'D:\code\python\voice coach\data\old_data\new_label_data_all\lm\mic\lm_staccatoE3_2'
wav_name = ['lm_staccatoE3_2_so_1_1.wav','lm_staccatoE3_2_so_1_3.wav','lm_staccatoE3_2_so_1_4.wav']
# file_list = os.listdir(test_dir_path)[:4]
file_list = [os.path.join(test_dir_path, name) for name in wav_name]
for file in file_list:
    file_path = os.path.join(test_dir_path, file)
    waveform, sample_rate = librosa.load(file_path, sr=48000, mono=True)
    print('shape', waveform.shape)
    print(sample_rate)
    for voice_segment in middle_cut(waveform, sample_rate, lens=300, pad_type='rand'):
        # 转换谱图的方法：建议mel谱
        print(file)
        print(voice_segment.shape)
        specgram = mel_specgram(voice_segment, sample_rate, n_mels=224, hop_length=112)

        # specgram = stft_specgram(voice_segment, sample_rate, n_fft=256, step=128)
        # max_freq = int((2*12000/sample_rate)*specgram.shape[0])
        # specgram = specgram[:max_freq,:]
        print(specgram.shape)
        plot_specgram(specgram)
        specgram = mel_specgram(voice_segment, sample_rate, n_mels=224, hop_length=112)
        plot_specgram(specgram)
        print(specgram.shape)