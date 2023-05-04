import torch

def pitch_shift(linear_spec, start, end, s):
    N = linear_spec.shape[0]  # the number of freq bins
    modified_spec = torch.zeros(linear_spec.shape)
    modified_spec[:,:start] = linear_spec[:, :start]
    modified_spec[:, end+1:] = linear_spec[:, end+1:]
    
    for t in range(start, end+1):
        for n in range(N):
            i = int(n /(2**(s/12)))
            if i < N:
                modified_spec[i, t]= modified_spec[i, t] + linear_spec[n, t]

    return modified_spec

