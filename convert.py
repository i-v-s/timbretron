import torch
import numpy as np
from torch_model import get_device, load_model
from nnAudio.Spectrogram import CQT
import pydub
from tqdm import tqdm


def main():
    device = get_device()
    cqt = CQT(48000, 1024, device=device)
    classes = 84 * 2
    model_name = 'wn1'
    model, _, _ = load_model(model_name, train=True, device=device)
    input_file = 'data/violin/i-s-bakh-partita-2-dlya-skripki-solo-re-minor.mp3'
    a = pydub.AudioSegment.from_mp3(input_file)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    y = y.astype(np.float32) / 32768
    window = 2 ** 18
    result = None
    with torch.no_grad():
        for offset in tqdm(range(0, len(y), window // 2)):
            chunk = y[offset: offset + window]
            t = cqt(torch.tensor(chunk, device=device).transpose(0, 1)).reshape(1, classes, -1)
            z = model(t).squeeze(0).transpose(0, 1)
            d = len(z) - window // 2
            z = (z[d // 2 : d // 2 - d] * 32768).cpu().numpy().astype(np.int16)
            result = np.concatenate((result, z), 0) if result is not None else z
    print(np.min(result), np.max(result))


if __name__ == '__main__':
    main()
