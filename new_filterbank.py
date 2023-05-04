import numpy as np
from matplotlib import pyplot as plt
import librosa
import torch
from scipy.io.wavfile import read

# from hz to mel scale
def hz_to_mel(frequencies, fmin, a, b):
    frequencies = np.asanyarray(frequencies)
    return a * np.log2((b * (frequencies/fmin)))

# from mel to hz 
def mel_to_hz(mels, fmin, a, b):
    mels = np.asanyarray(mels)
    return (2**(mels/a)) * fmin / b


# this is to obtain the triangle boundaries in hz, this generates n_mels + 2 points
def mel_frequencies(n_mels, fmin, fmax, a, b):
    min_mel = hz_to_mel(fmin, fmin, a, b)
    max_mel = hz_to_mel(fmax, fmin, a, b)
    mels = np.linspace(min_mel, max_mel, n_mels+2)

    return mel_to_hz(mels, fmin, a, b)


# this is to map the hz  boundaries to fft bin 
# multipy it by (n_fft + 1)/sr) and round it down to the nearest integer
def hz_to_bin(freqs, sr, n_fft):
    return np.floor(((n_fft + 1)/sr) * freqs).astype(int)


# this is to calculate the triangle weights.



def mel(bin, n_fft):
    n_mels = len(bin)-2
    weights = np.zeros((n_mels, int(1 + n_fft//2)))
    for m in range(1, n_mels + 1):
        f_m_minus = (bin[m - 1])   # left
        f_m = (bin[m])             # center
        f_m_plus = (bin[m + 1])    # right
        # print(f_m_minus)
        # print(f_m)
        # print(f_m_plus)
        for k in range(f_m_minus, f_m):
            weights[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            weights[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return weights



def plot(data, plot=True, saving_path="pic.png"):
    if plot == True:
        fig, (ax1, ax2) = plt.subplots(2) 
        ax1.plot(data.T)
        img =ax2.imshow(data, origin="lower", aspect = "auto")
        fig.colorbar(img)
        fig.savefig(saving_path)
    else:
        fig, ax1 = plt.subplots() 
        img =ax1.imshow(data, origin="lower", aspect = "auto")
        fig.colorbar(img)
        fig.savefig(saving_path)



if __name__ == "__main__":
    # mel_weight parameters
    n_fft = 2048
    n_mels = 80
    fmin = 60
    fmax = 11025 
    a = 12  # this is the a in the slide
    b = 1
    
   
   
    path = "/data1/ribka/waveglow/LJ001-0072.wav"
    sr, audio = read(path)
   
    boundaries = mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax, a=a, b=b)
    filter_points = hz_to_bin(boundaries, sr=sr, n_fft=n_fft)

    mel_weights = mel(filter_points, n_fft)   # modified mel weights
    print(boundaries)
    print(filter_points)
    print(mel_weights.shape)
    plot(mel_weights, saving_path ="weight_matrix.png")

    # testing

    hop_length = 256
    stft_args = {"n_fft":n_fft, "hop_length":hop_length}
    mel_args = {"sr":sr, "n_fft": n_fft, "n_mels": n_mels, "fmin": fmin, "fmax": fmax}

    magnitudes =torch.from_numpy(np.abs(librosa.stft(audio.astype(np.double), **stft_args)))
    new_mel_basis = torch.from_numpy(mel_weights).double()

    new_melspec= torch.matmul(new_mel_basis, magnitudes)
    new_melspec= torch.log(new_melspec)
    plot(new_melspec, plot=False, saving_path="new_melspec.png")
    print(mel(np.array([0,2,4,6,8,10]), 10))

    