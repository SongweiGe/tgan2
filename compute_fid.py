#!/usr/bin/env python3
# python3 scripts/compute_fid.py \
#     --ucf101-h5path-train /home/imaginaire/intern/songwei/imaginaire_intern/datasets/unit_test/raw/ucf101_h5_64x64/train.h5 \
#     --ucf101-config-train /home/imaginaire/intern/songwei/imaginaire_intern/datasets/unit_test/raw/ucf101_h5_64x64/train.json \
#     --c3d-pretrained-model ./models/conv3d_deepnetA_ucf.npz \
#     --stat-output ./ucf101_64px_stat.npz
import argparse
import os

import chainer
import chainer.cuda
import numpy
from chainer import Variable

import cv2
import h5py
import imageio
import pandas

import tgan2
import yaml
from tgan2 import C3DVersion1UCF101
from tgan2 import UCF101Dataset
from tgan2.evaluations import fid
from tgan2.evaluations import inception_score
from tgan2.utils import make_instance

import torch

# len(dset) * n_loops == 9537 * 10 == 5610 * 17
def get_mean_cov(classifier, dataset, batchsize=17, n_iterations=5610):
    N = len(dataset)
    xp = classifier.xp

    ys = []
    it = chainer.iterators.MultiprocessIterator(
        dataset, batch_size=batchsize, shuffle=False, repeat=True, n_processes=8)
    for i, batch in enumerate(it):
        if i == n_iterations:
            break
        print('Compute {} / {}'.format(i + 1, n_iterations))

        batch = chainer.dataset.concat_examples(batch)
        batch = Variable(xp.asarray(batch))  # To GPU if using CuPy

        # Feed images to the inception module to get the features
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = classifier.get_feature(batch)
        n_features = numpy.prod(y.shape[1:])
        ys.append(chainer.cuda.to_cpu(y.data.reshape(len(y.data), n_features)))

    # Compute mean and covariance
    ys = numpy.concatenate(ys)
    mean = numpy.mean(ys, axis=0)
    cov = numpy.cov(ys.T)
    return mean.astype(numpy.float32), cov.astype(numpy.float32)


def main():
    parser = argparse.ArgumentParser()

    # For calculating statistics as the preparation
    parser.add_argument(
        '--ucf101-h5path-train', type=str,
        default='datasets/ucf101_192x256/train.h5')
    parser.add_argument(
        '--ucf101-config-train', type=str,
        default='datasets/ucf101_192x256/train.json')
    parser.add_argument(
        '--c3d-pretrained-model', type=str,
        default='./models/conv3d_deepnetA_ucf.npz')
    parser.add_argument(
        '--stat-output', '-o', type=str,
        default='datasets/ucf101_192x256/ucf101_192x256_stat.npz')
    parser.add_argument('--test', action='store_true', default=False)

    # For calculating FID
    parser.add_argument('--stat-filename', '-s', type=str, default='./ucf101_64px_stat.npz')
    parser.add_argument(
        '--ucf101-h5path-test', type=str,
        default='datasets/ucf101_192x256/test.h5')
    parser.add_argument(
        '--ucf101-config-test', type=str,
        default='datasets/ucf101_192x256/test.json')
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--n-samples', '-n', type=int, default=2048)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--n-frames', '-f', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n-loops', type=int, default=10)
    parser.add_argument('--out', default='result.yml')

    args = parser.parse_args()

    if args.test:
        stat = numpy.load(args.stat_filename)
        fid_result = fid.get_FID(stat['mean'], stat['cov'], stat['mean'], stat['cov'])
        print('FID:', fid_result)
        exit()

    if args.stat_filename is None:
        print('Loading')
        dataset = UCF101Dataset(
            n_frames=args.n_frames,
            h5path=args.ucf101_h5path_train,
            config_path=args.ucf101_config_train,
            img_size=64,
            xmargin=0,
            stride=1)

        classifier = C3DVersion1UCF101(pretrained_model=args.c3d_pretrained_model)
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
            classifier.to_gpu()
        mean, cov = get_mean_cov(classifier, dataset)
        numpy.savez(args.stat_output, mean=mean, cov=cov)
    else:
        config = yaml.load(open(args.config))
        print(yaml.dump(config, default_flow_style=False))

        classifier = C3DVersion1UCF101(pretrained_model=args.c3d_pretrained_model)
        if args.gpu >= 0:
            classifier.to_gpu()
   
        scores = []
        stat = numpy.load(args.stat_filename)
        # video_paths = ['/home/imaginaire/intern/songwei/imaginaire_intern/logs/evaluation/training_samples/results/train/ode']
        # video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_700k/results/train/ode/',
        #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_900k/results/train/ode/',
        #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_reverse_diffusion_900k/results/train/ode/']
        # video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_400k/train/ode/',
        #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_500k/train/ode/',
        #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/2022_0610_0418_11_vp_ncsnpp_bs56_res64_infer_reverse_diffusion_500k/train/ode/',
        #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/ucf101_vp_ncsnpp_bs192_res64_infer_160k/results/train/ode/',
        #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/ucf101_vp_ncsnpp_bs96_res64_infer_150k/results/train/ode/',
        #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/ucf101_vp_ncsnpp_bs96_res64_infer_300k/results/train/ode/',
        #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_dropout0.2_infer_700k/results/train/ode/',
        #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_nogclip_infer_700k/results/train/ode/',
        #                '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_lr2e-4_infer_150k/results/train/ode/']
        # video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_infer_300k/results/train/ode/',
        #             '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_infer_700k/results/train/ode/',
        #             '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_infer_150k/results/train/ode/']
        video_paths = ['/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs24_res64_joint8_infer_300k/results/train/ode/',
                    '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs24_res64_joint8_infer_150k/results/train/ode/',
                    '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs32_res64_joint4_infer_150k/results/train/ode/',
                    '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs32_res64_joint4_infer_300k/results/train/ode/',
                    '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_joint4_infer_700k/results/train/ode/',
                    '/home/imaginaire/intern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_joint4_infer_150k/results/train/ode/',
                    '/home/imaginaire/i`ntern/songwei/workspace/results/video_diffusion/evaluation/vp_ncsnpp_bs48_res64_joint4_infer_300k/results/train/ode/']
        
        for video_path_base in video_paths: 
            model_name = video_path_base.split('/')[9]
            print(model_name)
            # for mode in ['average', 'regular']:
            for mode in ['average']:
                print(mode)
                video_path = os.path.join(video_path_base, mode)
                filenames = os.listdir(video_path)
                # import pdb;pdb.set_trace()
                # tmp = torch.load(os.path.join(video_path, filenames[0])).numpy()
                xs = [torch.load(os.path.join(video_path, fn)).numpy() for fn in filenames]
                xs = numpy.concatenate(xs, 0)
                # imageio.mimsave('debug64.mp4', make_video_grid(torch.from_numpy(tmp[:64])/255.), fps=2)
                xs = xs[:, :, :16]
                xs = xs.astype(numpy.float32)/255.*2-1
                scores = []
                mean, cov = fid.get_mean_cov(classifier, xs, args.batchsize)
                fid_result = fid.get_FID(stat['mean'], stat['cov'], mean, cov)
                print(fid_result)
                result = {'model_name': model_name, 'mode': mode, 'fid_score': str(fid_result)}
                open(args.out, 'a').write(yaml.dump(result, default_flow_style=False))

if __name__ == '__main__':
    main()
