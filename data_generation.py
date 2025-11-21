import numpy as np
import argparse, os

def generate_sine(n_series=1, length=1000, noise_std=0.1):
    t = np.arange(length)
    data = []
    for i in range(n_series):
        freq = 0.01 + 0.02 * np.random.rand()
        phase = 2*np.pi*np.random.rand()
        amp = 0.8 + 0.4*np.random.rand()
        series = amp * np.sin(2*np.pi*freq*t + phase) + 0.1*np.random.randn(length)
        data.append(series)
    return np.stack(data, axis=0)  # (n_series, length)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n_series', type=int, default=1)
    p.add_argument('--length', type=int, default=1000)
    p.add_argument('--save', type=str, default='results/data.npy')
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    arr = generate_sine(args.n_series, args.length)
    # Save as (time, features) for single-series usage
    # We store as (length, 1) when n_series==1, else first series only
    if args.n_series == 1:
        np.save(args.save, arr[0].reshape(-1,1))
    else:
        # save first series for demo
        np.save(args.save, arr[0].reshape(-1,1))
    print('Saved demo data to', args.save)
