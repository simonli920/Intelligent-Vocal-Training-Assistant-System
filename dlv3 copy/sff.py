import numpy as np


# 构建复指数矩阵
def _make_complex_factor(n_t, fs, K):
    def _f1(k):
        return np.pi * (1 - 2 * k / 48000)

    def _f2(nk):
        return np.exp(1j * nk)

    resolution = fs / 2 / K
    _a1 = _f1(np.arange(K) * resolution).reshape(1, -1)
    _a2 = np.arange(n_t).reshape(-1, 1)
    _a3 = _a2.dot(_a1)
    complex_factor = _f2(_a3)
    return complex_factor


def _SFF_filt(waveform, fs, K):
    def _f1(k):
        return np.pi * (1 - 2 * k / 48000)

    def _f2(nk):
        return np.exp(1j * nk)

    # 微分
    x = np.diff(waveform)
    N = x.shape[0]
    # 乘复指数
    x_nk = np.expand_dims(x, axis=1) * _SFF_COMPLEX_FACTOR[str(K)][:N, :K]
    # 单极滤波
    r = 0.995
    y_nk = x_nk.copy()
    for n in range(1, N):
        y_nk[n] += -r * y_nk[n - 1]
    # y_nk = signal.lfilter([1, 0], [1, r], x_nk, axis=0)
    # 取模
    v_nk = np.abs(y_nk)
    return v_nk.T


# 降采样谱图，取各频率在每个时间窗口的均值，默认行频率，列时间
def _downsample(spec, n_t):
    shift = int(spec.shape[1] / n_t + 1)
    ss_vec = np.arange(0, spec.shape[1], shift)
    spec_new = np.zeros((spec.shape[0], ss_vec.shape[0]))
    for i in range(ss_vec.shape[0] - 1):
        spec_new[:, i] = np.average(spec[:, ss_vec[i]:ss_vec[i + 1]], axis=1)
    spec_new[:, -1] = spec[:, -1]
    return spec_new


def SFF(waveform, fs, n_f=80, n_t=80, log=False):
    if str(n_f) not in _SFF_COMPLEX_FACTOR.keys():
        print('worng n_f, only support n_f at 40,80,120,160,200,224')
        raise
    full_spec = _SFF_filt(waveform, fs, n_f)
    if log:
        full_spec = 20 * np.log10(full_spec+1e-60)
    downsample_spec = _downsample(full_spec, n_t)
    return downsample_spec


def SFFCC(waveform, fs, n_f=80, n_t=80):
    nfft = n_f * 2
    log_sffspec = SFF(waveform, fs, n_f=n_f, n_t=n_t, log=True)
    log_sffspec_flip = np.flip(log_sffspec, axis=0)
    log_sffspec_full = np.r_[log_sffspec, log_sffspec_flip]
    _sffcc = np.fft.ifft(log_sffspec_full, n=nfft)


_SFF_COMPLEX_FACTOR = {}
_SFF_COMPLEX_FACTOR['40'] = _make_complex_factor(10000, 48000, 40)
_SFF_COMPLEX_FACTOR['80'] = _make_complex_factor(10000, 48000, 80)
_SFF_COMPLEX_FACTOR['120'] = _make_complex_factor(10000, 48000, 120)
_SFF_COMPLEX_FACTOR['160'] = _make_complex_factor(10000, 48000, 160)
_SFF_COMPLEX_FACTOR['200'] = _make_complex_factor(10000, 48000, 200)
_SFF_COMPLEX_FACTOR['224'] = _make_complex_factor(20000, 48000, 224)
# _SFF_COMPLEX_FACTOR['512'] = _make_complex_factor(12000, 48000, 512)







