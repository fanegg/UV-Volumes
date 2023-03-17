from lib.config import cfg, args


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    from lib.networks.renderer import make_renderer
    import os

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            batch['epoch'] = cfg.test.epoch
            output = renderer.render(batch)
            evaluator.evaluate_metrics(output, batch)
    metrics = evaluator.summarize()
    
    mse, psnr, ssim = metrics['zEval_mse'], metrics['zEval_psnr'], metrics['zEval_ssim']

    img_root = 'data/{}/{}/{}'.format(cfg.evaluate, cfg.task, cfg.exp_name)

    os.system('mkdir -p {}'.format(img_root))
    print('Eval results are saving to'+img_root)
    if cfg.use_lpips:
        lpips = metrics['zEval_lpips']
        with open(os.path.join(img_root, 'metrics.txt'), "w") as f:
            f.write(f'MSE: {mse:.6f}\nPSNR:{psnr:.3f}\nSSIM:{ssim:.3f}\nLPIPS:{lpips:.3f}')
    else:
        with open(os.path.join(img_root, 'metrics.txt'), "w") as f:
            f.write(f'MSE: {mse:.6f}\nPSNR:{psnr:.3f}\nSSIM:{ssim:.3f}')

if __name__ == '__main__':
    globals()['run_' + args.type]()
