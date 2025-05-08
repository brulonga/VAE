import torch    
import wandb
import numpy as np
from sklearn.manifold import TSNE
import os
import torch.distributed as dist
from Utils.loss import VAELoss
from Utils.plot import *
from Models._init_ import save_weights
from Utils.init_wandb import init_wandb
from Utils.setup_distributed import cleanup, clear_memory
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets.dataset import shuffle_sampler

def entrenamiento(opt, model, criterion, optimizer, scheduler, LOADER_ENTRENAMIENTO, samplers, rank, world_size):

    best_loss = 10000 
    num_epochs = opt['train']['epochs']
    log_interval = opt['train']['verbose']

    train_losses = []
    train_recon_losses = []
    train_KL_losses = []

    if opt['train']['loss'] == 'VAELoss':
        # Instanciamos la clase
        criterion = VAELoss(recon_loss_type='mse', beta_max=4.0, kl_annealing_epochs=50, free_bits=1.0, sigmoid_midpoint=50, sigmoid_steepness=10)
    else:
        raise ValueError(f"Función de pérdida no reconocida: {opt['train']['loss']}")

    if rank == 0:
        init_wandb(opt)

    for epoch in range(num_epochs):

        print('Epoca actual:', epoch)
        shuffle_sampler(samplers, epoch)
        print('Samplers distribuidos')
        
        model.train()  
        
        criterion.update_beta(25 + epoch)

        running_loss_train = 0.0
        running_recon_loss_train = 0.0
        running_KL_loss_train = 0.0

        for batch_idx, images in enumerate(LOADER_ENTRENAMIENTO):

            images = images.to(f'cuda:{rank}')

            optimizer.zero_grad()

            reconstruct_image, mean, log_variance = model(images)

            loss_train, recon_loss, kl_div, beta = criterion(reconstruct_image, images, mean, log_variance, epoch)

            loss_train.backward()

            optimizer.step()

            running_loss_train += loss_train.item()
            running_recon_loss_train += recon_loss.item()
            running_KL_loss_train += kl_div.item()

            global_step = epoch * len(LOADER_ENTRENAMIENTO) + batch_idx

            dist.barrier()

            if rank == 0:
                wandb.log({
                    'Batch Loss': loss_train.item(),
                    'Batch MSE': recon_loss.item(),
                    'Batch KL': kl_div.item(),
                    'Latent μ mean (0)':  mean.mean().item(),
                    'Latent μ std (1)':   mean.std().item(),
                    'Latent logvar mean (0)': log_variance.mean().item(),
                    'Latent logvar std (+-2)':  log_variance.std().item(),
                    'Latent σ² mean (1)': log_variance.exp().mean().item(),
                    'Latent σ² std (no 0 o inf)':  log_variance.exp().std().item(),
                    'beta': beta,
                }, step=global_step)

                if batch_idx % log_interval == 0:
                    for bi in range(min(2, images.size(0))):
                        inp = images[bi].permute(1,2,0).cpu().numpy().clip(0,1)
                        out = reconstruct_image[bi].permute(1,2,0).cpu().detach().numpy().clip(0,1)
                        wandb.log({
                            f"Input image {bi}":         wandb.Image(inp),
                            f"Reconstructed image {bi}": wandb.Image(out),
                        }, step=global_step)

        running_loss_train_tensor = torch.tensor(running_loss_train, dtype=torch.float32, device=f'cuda:{rank}')

        torch.distributed.reduce(running_loss_train_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)

        running_recon_loss_train_tensor = torch.tensor(running_recon_loss_train, dtype=torch.float32, device=f'cuda:{rank}')

        torch.distributed.reduce(running_recon_loss_train_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)

        running_KL_loss_train_tensor = torch.tensor(running_KL_loss_train, dtype=torch.float32, device=f'cuda:{rank}')

        torch.distributed.reduce(running_KL_loss_train_tensor, dst=0, op=torch.distributed.ReduceOp.SUM)

        if rank == 0:
            running_loss_train = running_loss_train_tensor.item()
            running_recon_loss_train = running_recon_loss_train_tensor.item()
            running_KL_loss_train = running_KL_loss_train_tensor.item()

        dist.barrier()

        if rank == 0:

            avg_train_loss = running_loss_train / len(LOADER_ENTRENAMIENTO)

            train_losses.append(avg_train_loss)

            avg_train_recon_loss = running_recon_loss_train / len(LOADER_ENTRENAMIENTO)

            train_recon_losses.append(avg_train_recon_loss)

            avg_train_KL_loss = running_KL_loss_train / len(LOADER_ENTRENAMIENTO)

            train_KL_losses.append(avg_train_KL_loss)

            perdida_entrenamiento = running_loss_train / len(LOADER_ENTRENAMIENTO)

            print(f"Época [{epoch+1}/{num_epochs}], Perdida: {perdida_entrenamiento:.4f}, Perdida MSE: {avg_train_recon_loss:.4f}, Perdida KL divergence: {avg_train_KL_loss:.4f}")

            if perdida_entrenamiento < best_loss:
                best_loss = perdida_entrenamiento
                save_weights(
                    model,
                    optimizer,
                    scheduler,
                    opt['network']['save_weights'],  # p.ej. 'checkpoint_best.pth'
                    rank=0
                )
                print("Mejor modelo guardado")

            last_path = opt['network']['save_weights'].replace('_best', '_last')
            save_weights(
                model,
                optimizer,
                scheduler,
                last_path,           # p.ej. 'checkpoint_last.pth'
                rank=0
            )
            print(f"Pesos de la última época guardados en {last_path}")

        dist.barrier()

        clear_memory()

        # if (epoch+1)%3 == 0:

        #     model.eval()

        #     all_mu = []
        #     with torch.no_grad():
        #         for batch_idx, images in LOADER_ENTRENAMIENTO:
        #             images = images.to(f'cuda:{rank}')
        #             encoder = model.module.encoder if hasattr(model, 'module') else model.encoder
        #             mu, logvar = encoder(images)
        #             all_mu.append(mu.cpu())

        #             samples_per_proc = 4000 // world_size
        #             if sum(t.size(0) for t in all_mu) >=samples_per_proc:
        #                 break

            
        #     local_latents = torch.cat(all_mu, dim=0)[:samples_per_proc]  # (max_samples, latent_dim)

        #     gather_list = [torch.zeros_like(local_latents) for _ in range(world_size)]
        #     dist.all_gather(gather_list, local_latents.cuda(rank))

        #     if rank == 0:
                
        #         all_latents = torch.cat(gather_list, dim=0).cpu().numpy()  # shape: (world_size*max_samples, latent_dim)

        #         save_dir = '/home/brulon/VAE/latents'
        #         file_path = os.path.join(save_dir, f'latents_epoch_{epoch:03d}.pt')
        #         torch.save(all_latents, file_path)
        #         tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
        #         X2 = tsne.fit_transform(all_latents)

        #     dist.barrier()

        #     clear_memory()

    cleanup()

    return perdida_entrenamiento

