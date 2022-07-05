#!/usr/bin/env python3

import argparse
import warnings

import chainer
from chainer import Variable
import numpy
import yaml
import math
import numpy as np

import tgan2
import tgan2.evaluations.inception_score
from tgan2.utils import make_config
from tgan2.utils import make_instance
from tgan2 import UCF101Dataset

import os
import torch
import imageio


def get_real_mean_std(dataset, batchsize=20, n_iterations=500):
    N = len(dataset)

    ys = []
    it = chainer.iterators.MultiprocessIterator(
        dataset, batch_size=batchsize, shuffle=False, repeat=True, n_processes=8)
    for i, batch in enumerate(it):
        if i == n_iterations:
            break
        batch = chainer.dataset.concat_examples(batch)
        # Feed images to the inception module to get the features
        ys.append(batch)

    ys = numpy.concatenate(ys)
    # Compute mean and covariance
    return ys

def make_video_grid(video, nrow=None, padding=1):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1)
    video = (video.cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                        (padding + w) * ncol + padding, c), dtype='uint8')
    print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    return video
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'infiles', nargs='+', type=argparse.FileType('r'), default=())
    parser.add_argument('-a', '--attrs', nargs='*', default=())
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-m', '--model', default='arg.npz')
    parser.add_argument('-o', '--out', default='result.yml')
    parser.add_argument('--n-samples', type=int, default=2048)
    parser.add_argument('--n-loops', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=5)
    parser.add_argument('--n-frames', type=int, default=16)
    args = parser.parse_args()

    conf_dicts = [yaml.load(fp, yaml.Loader) for fp in args.infiles]
    config = make_config(conf_dicts, args.attrs)
    return config, args


def main(config, args):
    # print('Prepare model: {}'.format(args.model))
    # gen = make_instance(
    #     tgan2, config['gen'], args={'out_channels': 3})
    # chainer.serializers.load_npz(args.model, gen)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        # gen.to_gpu()

    conf_classifier = config['inception_score']['classifier']
    classifier = make_instance(tgan2, conf_classifier)
    if 'model_path' in conf_classifier:
        chainer.serializers.load_npz(
            conf_classifier['model_path'],
            classifier, path=conf_classifier['npz_path'])
    if args.gpu >= 0:
        classifier.to_gpu()


    # video_paths = ['/home/imaginaire/intern/songwei/workspace/projects/imaginaire/logs/2022_0610_0422_16_vp_ncsnpp_bs128_res32_infer/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/projects/imaginaire/logs/2022_0610_0444_24_vp_ncsnpp_bs48_res64_nogclip_infer/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/projects/imaginaire/logs/2022_0610_0452_25_vp_ncsnpp_bs48_res64_dropout0.2_infer/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/projects/imaginaire/logs/2022_0610_0458_12_vp_ncsnpp_bs48_res64_dropout0_infer/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/projects/imaginaire/logs/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/projects/imaginaire/logs/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_15k/train/ode/']
    # video_paths = ['/home/imaginaire/intern/songwei/workspace/projects/imaginaire/logs/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_40k/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/projects/imaginaire/logs/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_50k/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/projects/imaginaire/logs/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_reverse_diffusion_50k/train/ode/']
    # video_paths = ['/home/imaginaire/intern/songwei/imaginaire_intern/logs/evaluation/training_samples/results/train/ode']
    # video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_700k/results/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_900k/results/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_reverse_diffusion_900k/results/train/ode/']
    # video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/ucf101_vp_ncsnpp_bs192_res64_infer_160k/results/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/ucf101_vp_ncsnpp_bs96_res64_infer_150k/results/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/ucf101_vp_ncsnpp_bs96_res64_infer_300k/results/train/ode/']
    # video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_dropout0.2_infer_700k/results/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_nogclip_infer_700k/results/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_lr2e-4_infer_150k/results/train/ode/']
    video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs24_res64_joint8_infer_300k/results/train/ode/',
                   '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs24_res64_joint8_infer_300k/results/train/ode/',
                   '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs24_res64_joint8_infer_150k/results/train/ode/',
                   '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs32_res64_joint4_infer_150k/results/train/ode/',
                   '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs32_res64_joint4_infer_300k/results/train/ode/',
                   '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_joint4_infer_700k/results/train/ode/',
                   '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_joint4_infer_150k/results/train/ode/',
                   '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_joint4_infer_300k/results/train/ode/']
    # video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_infer_300k/results/train/ode/',
    #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_infer_700k/results/train/ode/']
    # video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluWation/vp_ncsnpp_bs56_res64_infer_pc_euler_100step_900k/results/train/pc']
    video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/multi_node/ucf101_vp_adm_bs64_uncond_res64_150k/results/train/ode/']
    # Real data IS
    # dataset = UCF101Dataset(
    #     n_frames=16,
    #     h5path='/home/imaginaire/intern/songwei/imaginaire_intern/datasets/unit_test/raw/ucf101_h5_64x64/train.h5',
    #     config_path='/home/imaginaire/intern/songwei/imaginaire_intern/datasets/unit_test/raw/ucf101_h5_64x64/train.json',
    #     img_size=64,
    #     xmargin=0,
    #     stride=1)
    # test_dataset = UCF101Dataset(
    #     n_frames=16,
    #     h5path='/home/imaginaire/intern/songwei/imaginaire_intern/datasets/unit_test/raw/ucf101_h5_64x64/test.h5',
    #     config_path='/home/imaginaire/intern/songwei/imaginaire_intern/datasets/unit_test/raw/ucf101_h5_64x64/test.json',
    #     img_size=64,
    #     xmargin=0,
    #     stride=1)
    # xs = get_real_mean_std(dataset)
    # test_xs = get_real_mean_std(test_dataset, n_iterations=500)
    # mean, std = tgan2.evaluations.inception_score.inception_score(classifier, np.concatenate([xs[:8000], test_xs]), args.batchsize, splits=1)
    # mean, std = tgan2.evaluations.inception_score.inception_score(classifier, test_xs, args.batchsize, splits=1)

    for video_path_base in video_paths: 
        model_name = video_path_base.split('/')[9]
        print(model_name)
        # for mode in ['average', 'regular']:
        for mode in ['average']:
            print(mode)
            video_path = os.path.join(video_path_base, mode)
            filenames = os.listdir(video_path)
            import pdb;pdb.set_trace()
            # tmp = torch.load(os.path.join(video_path, filenames[0])).numpy()
            # imageio.mimsave('debug64.mp4', make_video_grid(torch.from_numpy(tmp[:64])/255.), fps=2)
            xs = [torch.load(os.path.join(video_path, fn)).numpy() for fn in filenames]
            xs = np.concatenate(xs, 0)
            xs = xs[:, :, :16]
            xs = xs.astype(np.float32)/255.*2-1
            imageio.mimsave('debug64.mp4', make_video_grid(torch.from_numpy(xs[:160])/255., nrow=8), fps=2)
            scores = []
            mean, std = tgan2.evaluations.inception_score.inception_score(classifier, xs, args.batchsize, splits=1)
            print(f'{mean} +- {std}')
            result = {'model_name': model_name, 'mode': mode, 'mean': str(mean), 'std': str(std)}
            # imageio.mimsave('debug64_train.mp4', make_video_grid(torch.from_numpy(xs[:256])/2+0.5), fps=4)
            open(args.out, 'a').write(yaml.dump(result, default_flow_style=False))


if __name__ == '__main__':
    config, args = parse_args()
    # Ignore warnings
    warnings.simplefilter('ignore')
    main(config, args)
