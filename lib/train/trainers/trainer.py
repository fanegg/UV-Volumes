import time
import torch
import tqdm
from lib.config import cfg
import os
import imageio
import cv2
import numpy as np
from lib.networks import embedder


class Trainer(object):
    def __init__(self, network):
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            else:
                batch[k] = batch[k].to(self.device)
        return batch

    def train(self, epoch, data_loader, optimizer, recorder, ep_tqdm):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = self.to_cuda(batch)

            batch['epoch'] = epoch
            output, loss, loss_stats, image_stats = self.network(batch, is_train=True)

            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40) 
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)
            lr = {'lr2': optimizer.param_groups[0]['lr'],
                  'lr1': optimizer.param_groups[-1]['lr']}
            recorder.update_lr_stats(lr)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.record_interval == 0 or iteration == (max_iter - 1):
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)
        latent_index_record = []


        for batch in tqdm.tqdm(data_loader):
            batch = self.to_cuda(batch)
            batch['epoch'] = epoch
            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch, is_train=False)
                if evaluator is not None:
                    evaluator.evaluate(output, batch, epoch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

            # texture map
            if batch['latent_index'] in latent_index_record:
                continue

            with torch.no_grad():
                tex_size = cfg.texture_size
                i_onehot = torch.eye(24)[torch.arange(24)].cuda().unsqueeze(1).expand(-1,tex_size*tex_size,-1) #24,256*256,24
                umap, vmap = torch.meshgrid(torch.linspace(0,1,tex_size).to(self.device), 
                                                torch.linspace(0,1,tex_size).to(self.device))
                uv_stack = torch.stack((umap, vmap), 2).view(-1,1,2)   # 256*256,1,2

                uv_encoding = embedder.uv_embedder(uv_stack.view(1,-1,2)) # 1, mask, 42
                uv_encoding = uv_encoding.expand(24,-1,-1)  #  24, mask, 42
                iuv_encoding = torch.cat((i_onehot, uv_encoding), -1)  # 24, mask, 24 + 42
                iuv_encoding = iuv_encoding.view(-1, iuv_encoding.shape[-1]) # 2 * mask, 24+42

                expand_view = torch.Tensor([1,0,0]).to(self.device)[None,None].expand(24,tex_size*tex_size,-1).view(-1, 3)
                viewdirs_encoding = embedder.view_embedder(expand_view)   # top_k * mask, 27
                rgb_pred = self.network.net.implicit_tex_model.get_rgb(iuv_encoding, batch['poses'], viewdirs_encoding)
                Texture_pose = rgb_pred.view(24,tex_size,tex_size,3).cpu().numpy() * 255

                TextureIm_pose = np.zeros((tex_size * 4, tex_size * 6, 3), dtype=np.uint8)
                for i in range(len(Texture_pose)):
                    x = i // 6 * tex_size
                    y = i % 6 * tex_size
                    TextureIm_pose[x:x + tex_size, y:y + tex_size] = Texture_pose[i]     

                result_dir = os.path.join(cfg.result_dir, 'comparison')
                os.system('mkdir -p {}'.format(result_dir))
                frame_index = batch['frame_index'].item()
                cv2.imwrite(
                    '{}/texture_static_frame{:04d}_epoch{:04d}.png'.format(result_dir, frame_index, epoch), 
                    TextureIm_pose[..., [2, 1, 0]])

                # view rgb
                rgbs = []
                alpha = np.linspace(-np.pi, np.pi, 6)
                beta = np.linspace(-np.pi, np.pi, 6)
                for a in alpha:
                    for b in beta:
                        x = np.cos(a) * np.cos(b)
                        z = np.sin(a) * np.cos(b)
                        y = np.sin(b)
                        viewdir = torch.Tensor([x,y,z])[None,None].expand(24, tex_size*tex_size, -1).to(self.device) #24, 256*256, 3

                        viewdirs_encoding = embedder.view_embedder(viewdir.view(-1, 3))
                        rgb_pred = self.network.net.implicit_tex_model.get_rgb(iuv_encoding, batch['poses'], viewdirs_encoding)
                        rgb_dynamic = rgb_pred.view(24,tex_size,tex_size,3)

                        Texture = rgb_dynamic.cpu().numpy() * 255   #24, 256, 256, 3
                        TextureIm = np.zeros((tex_size * 4, tex_size * 6, 3), dtype=np.uint8)
                        for i in range(len(Texture)):
                            x = i // 6 * tex_size
                            y = i % 6 * tex_size
                            TextureIm[x:x + tex_size, y:y + tex_size] = Texture[i]

                        rgbs.append(TextureIm)

                imageio.mimsave(
                    '{}/texture_dynamic_frame{:04d}_epoch{:04d}.gif'.format(result_dir, frame_index, epoch), rgbs, fps=5)

            latent_index_record.append(batch['latent_index'])


        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
