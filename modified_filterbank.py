import numpy as np
import librosa 
from matplotlib import pyplot as plt
import torch
from scipy.io.wavfile import read
import warnings

# def hz_to_mel(frequencies, fmin, a, b):
#     return 2595.0 * np.log10(1.0 + frequencies / 700.0)


# def mel_to_hz(mels, fmin, a, b):
#     return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

# from hz to mel scale
def hz_to_mel(frequencies, fmin, a, b):
    frequencies = np.asanyarray(frequencies)
    return a * np.log2((b * (frequencies/fmin)))

# from mel to hz 
def mel_to_hz(mels, fmin, a, b):
    return (2**(mels/a)) * fmin / b

def mel_frequencies(n_mels, fmin, fmax, a, b):
    min_mel = hz_to_mel(fmin, fmin, a, b)
    #max_mel = np.ceil(hz_to_mel(fmax, fmin, a, b))
    max_mel = hz_to_mel(fmax, fmin, a, b)
    # print(min_mel)
    # print(max_mel)
    mels = np.linspace(min_mel, max_mel, n_mels)
    # print(mels)
    return mel_to_hz(mels, fmin, a, b)


def fft_frequencies(sr, n_fft):
    return np.fft.rfftfreq(n=n_fft, d = 1.0/sr)


def fft_frequencies_mine(sr, n_fft):
    return np.arange(0, 1 + n_fft / 2) * sr / n_fft


def mel(sr, n_fft, n_mels, fmin, fmax, a, b):
    weights = np.zeros((n_mels, int(1 + n_fft//2)))
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    mel_f = mel_frequencies(n_mels+2, fmin=fmin, fmax=fmax, a=a, b=b)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    
    # enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    # weights *= enorm[:, np.newaxis]

    # if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
    #     # This means we have an empty channel somewhere
    #     warnings.warn(
    #         "Empty filters detected in mel frequency basis. "
    #         "Some channels will produce empty responses. "
    #         "Try increasing your sampling rate (and fmax) or "
    #         "reducing n_mels.",
    #         stacklevel=2,
    #     )
    
    return weights


def mel(filter_points, n_fft):
    n_mels = len(filter_points)-2
    weights = np.zeros((n_mels, int(1 + n_fft//2)))

    for i in range(n_mels):
        interval_left = filter_points[i+1] - filter_points[i]
        interval_right = filter_points[i+2] - filter_points[i+1]
        weights[i, filter_points[i]:filter_points[i+1]] = np.linspace(0, 1, interval_left) # this creates the weights
        weights[i, filter_points[i+1]:filter_points[i+2]] = np.linspace(1, 0, interval_right)
    
    return weights

def plot(data):
    plt.plot(data)
    plt.savefig("triangle.png")


def plot_two_subplots(data1, data2, saving_path):
    fig, (ax1, ax2) = plt.subplots(2, sharex = True) 
    img1 = ax1.imshow(data1, aspect= "auto", origin="lower")
    ax1.set_title("melspectrogram using the original filterbank") 
    ax1.set_ylabel("Freq")
    img2 = ax2.imshow(data2, aspect= "auto",origin="lower")
    ax2.set_title("melspectrogram using the new filterbank")
    ax2.set_xlabel("Time(frames)")
    ax2.set_ylabel("Freq")
    fig.colorbar(img1, ax=fig.get_axes())
    fig.savefig(saving_path, bbox_inches='tight')  



if __name__ == "__main__":
    # mel_weight parameters
    n_fft = 2048
    n_mels = 80
    fmin = 60
    fmax = 11025 
    a = 12
    b = 1
    
    # stft_parameter
    hop_length = 256
    stft_args = {"n_fft":n_fft, "hop_length":hop_length}
   
    path = "/data1/ribka/waveglow/LJ001-0072.wav"
    sr, audio = read(path)
    mel_args = {"sr":sr, "n_fft": n_fft, "n_mels": n_mels, "fmin": fmin, "fmax": fmax}
    # boundaries = mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax, a=a, b=b)
    # filter_points = hz_to_bin(boundaries, sr=sr, n_fft=n_fft)

    mel_weights = mel(sr, n_fft, n_mels, fmin, fmax, a, b)   # modified mel weights
    mel_lib = librosa.filters.mel(**mel_args, htk=True, norm = None)
    plot(mel_weights)
    # print(mel_weights.shape)


    # testing
    magnitudes =torch.from_numpy(np.abs(librosa.stft(audio.astype(float), **stft_args)))

    original_mel_basis = torch.from_numpy(librosa.filters.mel(**mel_args)).double() # original mel weights
    new_mel_basis = torch.from_numpy(mel_weights).double()

    original_melspec= torch.matmul(original_mel_basis, magnitudes)
    new_melspec= torch.matmul(new_mel_basis, magnitudes)

    original_melspec = torch.log(original_melspec) # put the magnitude into log
    new_melspec= torch.log(new_melspec)
    plot()
    plot_two_subplots(mel_weights, mel_lib, "comparison.png")
 
   