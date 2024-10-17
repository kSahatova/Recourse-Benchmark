import numpy as np
import pandas as pd
import os.path as osp

from typing import List
from tqdm import tqdm 

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from carla_visual.models.utils import fetch_weights, load_model_weights
from carla_visual.models.autoencoder import AE

from carla_visual.dataloaders.online_dataset import load_online_dataset, build_dataloader
from carla_visual.plotting.plot_output import plot_image


AE_WEIGHTS_URLS = ['https://github.com/EoinKenny/AAAI-2021/raw/refs/heads/master/weights/ae_0.pth',
                'https://github.com/EoinKenny/AAAI-2021/raw/refs/heads/master/weights/ae_1.pth',
                'https://github.com/EoinKenny/AAAI-2021/raw/refs/heads/master/weights/ae_2.pth',
                'https://github.com/EoinKenny/AAAI-2021/raw/refs/heads/master/weights/ae_3.pth',
                'https://github.com/EoinKenny/AAAI-2021/raw/refs/heads/master/weights/ae_4.pth',
                'https://github.com/EoinKenny/AAAI-2021/raw/refs/heads/master/weights/ae_5.pth',
                'https://github.com/EoinKenny/AAAI-2021/raw/refs/heads/master/weights/ae_6.pth',
                'https://github.com/EoinKenny/AAAI-2021/raw/refs/heads/master/weights/ae_7.pth',
                'https://github.com/EoinKenny/AAAI-2021/raw/refs/heads/master/weights/ae_8.pth',
                'https://github.com/EoinKenny/AAAI-2021/raw/refs/heads/master/weights/ae_9.pth']


def train_autoencoders_mnist(train_data: Dataset, num_classes: int, 
                             num_epochs, output_dir: str = '') -> List[nn.Module]:
    ae_models_list = []
    for i in tqdm(range(num_classes)):

        model = AE()

        target_class = i
        learning_rate = 0.001
        output_file = osp.join(output_dir, f'ae_{target_class}.pth')

        train_indices = torch.nonzero(train_data.targets==target_class)
        train_images = train_data.data[train_indices].float()
        class_dataloader = DataLoader(train_images, batch_size=batch_size)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        _, model = train_autoencoder(model, class_dataloader, num_epochs, optimizer, 
                                     criterion=torch.nn.MSELoss(),
                                     logging_interval=num_epochs, skip_epoch_stats=False, 
                                     save_model_to=output_file)
        ae_models_list.append(model)
    return ae_models_list


def compute_epoch_loss_autoencoder(model, train_loader, criterion):
    total_loss = 0
    avg_loss = 0
    n_batches = 0

    for data in train_loader:
        recon = model(data)
        loss = criterion(data, recon)
        total_loss += loss
        n_batches += 1
    
    avg_loss = total_loss / n_batches

    return avg_loss


def train_autoencoder(model, train_loader, num_epochs, optimizer, criterion,
                      logging_interval, skip_epoch_stats, save_model_to):
    log_dict = {'train_loss_per_batch': [],
                'train_loss_per_epoch': []}
    
    if criterion is None:
        criterion = F.mse_loss

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, data in tqdm(enumerate(train_loader)):
            # FORWARD AND BACK PROP
            recon_data = model(data)
            loss = criterion(recon_data, data)

            optimizer.zero_grad()
            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            # if not batch_idx % logging_interval:
            #     print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
            #           % (epoch+1, num_epochs, batch_idx,
            #               len(train_loader), loss))
        
        if not skip_epoch_stats:
            model.eval()
            with torch.set_grad_enabled(False):  # save memory during inference
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, criterion)
                print('***Epoch: %03d/%03d | Loss: %.3f' % (
                        epoch+1, num_epochs, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())

    if save_model_to is not None:
        torch.save(model.state_dict(), save_model_to)
    
    return log_dict, model


def evaluate_autoencoder(model: nn.Module, dataloader: DataLoader, 
                         batch_size : int = 64, device: str='cpu'):
    total_mse = 0
    n_batches = dataloader.dataset.shape[0] // batch_size

    criterion = torch.nn.MSELoss(reduction='mean')
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            recon = model(data)
            #print(data.cpu().numpy().flatten().shape, recon.cpu().numpy().flatten().shape)

            # Calculate MSE and MAE
            mse = criterion(data, recon)
            total_mse += mse

    avg_mse = total_mse / n_batches
    print(f"Average MSE: {avg_mse:.4f}")

    return avg_mse


if __name__ == "__main__":    
    TRAIN_AES = False
    FETCH_AE_WEIGHTS = False
    TRAIN_COMPLETE_AE = True

    output_dir = r'D:\PycharmProjects\XAIRobustness\CARLA_VISUAL\carla_visual\evaluation\interpretability_metrics_looveren'
    weights_folder = 'ae_weights_mnist' # 'trained_ae_weights_mnist' if TRAIN_AES else 'ae_weights_mnist'

    num_classes = 10 
    num_epochs = 30
    ds_name = 'MNIST'
    data_root = 'D:\PycharmProjects\XAIRobustness\data\images'
    batch_size = 256

    train_data, test_data = load_online_dataset(ds_name, data_root, download=False)


    if TRAIN_COMPLETE_AE:
        model = AE()
        train_dataloader = DataLoader(train_data.data.float().unsqueeze(-1).permute(0, 3, 1, 2), batch_size=batch_size)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        train_autoencoder(model, train_dataloader, num_epochs, optimizer, criterion,
                          logging_interval=50, skip_epoch_stats=False, save_model_to=osp.join(output_dir, 'ae_full.pth'))
        
    elif TRAIN_AES:
        ae_models_list = train_autoencoders_mnist(train_data, num_classes, num_epochs, 
                                                  output_dir=osp.join(output_dir, weights_folder))
    else:
        if FETCH_AE_WEIGHTS:           
            for i, ae_url in enumerate(AE_WEIGHTS_URLS):
                fetch_weights(ae_url, file_name=f'ae_{i}.pth', output_dir=osp.join(output_dir, weights_folder))
        
        ae_models_list = [load_model_weights(AE(), osp.join(output_dir, weights_folder, f'ae_{i}.pth'), device='cpu') 
                            for i in range(num_classes)]

        print('Check the quality of the trained autoencoders')   

        
        for i, ae_model in enumerate(ae_models_list): 
            indices = torch.nonzero(test_data.targets==i)
            images = test_data.data[indices].float()
            class_dataloader = DataLoader(images, batch_size=batch_size)
            print(f'Reconstruction error of the AE for the class {i}')
            evaluate_autoencoder(ae_model, class_dataloader, batch_size=batch_size)

    
            