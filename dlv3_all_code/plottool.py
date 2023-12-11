import scipy.signal as signnal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def wav2np(path, one_chanel=False, normalization=False):
    import numpy as np
    import wave
    wave_read = wave.open(path, "rb")
    params = wave_read.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = wave_read.readframes(nframes)
    wave_read.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, nchannels
    wave_data = wave_data.T
    if one_chanel == True:
        wave_data = wave_data[0]
    if not normalization:
        return wave_data, framerate
    else:
        return wave_data / 32767, framerate


def plot_specgram(specgram):
    # LogarithmicSpectrogramData=np.abs(y)*np.abs(y)
    plt.pcolormesh(specgram)
    plt.colorbar()
    # plt.savefig('语谱图22.png')
    plt.show()


def plot_wave_spec(waveform, specgram):
    # 创建画布
    fig1, ax = plt.subplots(1, 2, figsize=(11,4))
    ax = ax.flatten()
    ax[0].plot(waveform)
    ax1 = ax[1].pcolormesh(specgram)
    fig1.colorbar(ax1)
    plt.tight_layout()

    plt.show()


def wav2np(path,one_chanel = False, normalization=False):
    import numpy as np
    import wave
    wave_read = wave.open(path, "rb")
    params = wave_read.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = wave_read.readframes(nframes)
    wave_read.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, nchannels
    wave_data = wave_data.T
    if one_chanel == True:
        wave_data = wave_data[0]
    if not normalization:
        return wave_data,framerate
    else:
        return wave_data/32767,framerate


# 绘制混淆矩阵
def plot_confusion_matrix(real_value, pred_value, n_classes=5,
                          savename='confusion_matrix.png', title='Confusion Matrix'):
    classes = [str(i) for i in range(n_classes)]
    cm = confusion_matrix(real_value, pred_value)
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    # plt.show()

# path=r'D:\code\python\voice coach\data\raw_datasetV2\lm\mic\lm_staccatoE3_2\lm_staccatoE3_2_mi_1_2.wav'
# voice_array,sample_rate = wav2np(path,one_chanel=True)
# lens_voice = len(voice_array)
# lens_win = int(200*sample_rate/1000)
# if lens_voice < lens_win:
#     voice_array = np.pad(voice_array, (0, lens_win-lens_voice), 'constant', constant_values=(0, 0))
# print(voice_array.shape)
# f,t,y = signnal.stft(voice_array, sample_rate,padded=False)
# y=10*np.log10((np.abs(y)*np.abs(y))) #取对数后的数据
# print(y.shape)
# plot_wave_spec(voice_array,y)