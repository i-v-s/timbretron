import torch
from torch_model import get_device, load_model, save_model, create_optimizer
from torch_model.data import CachedAudioDataset
from torch.utils.data import DataLoader
from nnAudio.Spectrogram import CQT
from torch.nn import MSELoss
from tqdm import tqdm
from sys import stdout


def main(batch_size=8):
    device = get_device()
    ds = CachedAudioDataset('data/violin', 2 ** 18)
    dl = DataLoader(ds, batch_size, True)
    cqt = CQT(48000, 1024, device=device)
    classes = 84 * 2
    model_name = 'wn1'
    model, best_loss, epoch = load_model(model_name, train=True, device=device)
    optimizer = create_optimizer({'type': 'adam', 'lr': 1e-4}, model)
    loss_func = MSELoss()
    while True:
        epoch += 1
        for train in [True]:
            if train:
                model.train()
            else:
                model.eval()
            count, loss_sum, worst_loss = 0, 0, 0
            for x in tqdm(dl):
                if len(x) < batch_size:
                    continue
                x = x.permute(0, 2, 1).to(device)
                t = cqt(x.reshape(batch_size * 2, -1)).reshape(batch_size, classes, -1)
                optimizer.zero_grad()
                y = model(t)
                d = x.shape[-1] - y.shape[-1]
                loss = loss_func(y, x[..., d // 2 : d // 2 - d])
                loss_sum += loss.item()
                count += 1
                if train:
                    loss.backward()
                    optimizer.step()
        save_model(model_name, model, best_loss, epoch)
        stdout.write(f'\nEpoch {epoch}, mean loss is {loss_sum / count if count > 0 else "?"}, worst loss is {worst_loss}\n')


if __name__ == '__main__':
    main()

