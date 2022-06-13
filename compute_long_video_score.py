#!/usr/bin/env python3

import argparse
import warnings

import os
import chainer
import numpy
import yaml
import numpy as np

import tgan2
import tgan2.evaluations.inception_score
from tgan2.utils import make_config
from tgan2.utils import make_instance


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

    # import pdb; pdb.set_trace()
    long_tplt = '/fs/vulcan-projects/contrastive_learning_songweig/TATS/results/numpy_files/ucf101_long_1024/cond_gpt_ucf_128_488_29999_epoch=8-step=579999-train_spa_16_temp_256_3_topp0.92_topk2048_run%d_eval_fastoneshot.npy'
    long_tplt = '/fs/vulcan-projects/contrastive_learning_songweig/TATS/results/numpy_files/ucf101_long_1024/cond_gpt_ucf_128_488_29999_epoch=8-step=579999-train_spa_16_temp_256_1_topp0.92_topk2048_run%d_eval_fastoneshot.npy'
    long_tplt = '/fs/vulcan-projects/contrastive_learning_songweig/TATS/results/numpy_files/ucf101_long_1024/cond_gpt_ucf_128_488_zero_23999_epoch=8-step=559999-train_spa_16_temp_256_1_topp0.92_topk2048_run%d_eval_fastoneshot.npy'
    long_tplt = '/fs/vulcan-projects/contrastive_learning_songweig/TATS/baselines/mocogan-hd/results/ucf101/1024frames_run%d.npy'
    long_tplt = '/fs/vulcan-projects/contrastive_learning_songweig/TATS/results/numpy_files/ucf101_hierarchical_long_1024/hierarchical_vqgan_cond_gpt_hierarchical_64_16_ucf_128_488_29999_hierarchical_vqgan_cond_gpt_hierarchical_20_infill_ucf_128_488_29999_topp0.92_topk2048_run%d_eval_fastoneshot.npy'
    long_tplt = '/fs/vulcan-projects/contrastive_learning_songweig/TATS/baselines/digan/results/ucf-101-train-test1024frames_%d.npy'
    # long_tplt = '/fs/vulcan-projects/contrastive_learning_songweig/TATS/baselines/ccvs/results/bak'
    # long_files = os.listdir(long_tplt)
    ys_all = []
    for j in range(5):
    # for j in range(1, 12):
        xs = np.load(long_tplt%j)
        # xs = np.load(os.path.join(long_tplt, long_files[j], '256frames.npy'))
        # import pdb;pdb.set_trace()
        scores = []
        ys_j = tgan2.evaluations.inception_score.compute_logits(classifier, xs[:, :1024], args.batchsize, splits=1)
        # ys_j = tgan2.evaluations.inception_score.compute_logits(classifier, xs, args.batchsize, splits=1)
        ys_all.append(ys_j)

    ys_np = np.concatenate(ys_all, 1) # 64 x 550 x 101
    labels = np.argmax(ys_np, -1) # 64 x 550
    y_eq = np.mean((labels == labels[0]).astype(np.float32), 1) # 550 x 64

    eps=1e-20
    ys_np_eps = ys_np + eps
    kl = (ys_np_eps * (np.log(ys_np_eps) - np.log(ys_np_eps[0]))).sum(-1)
    kl_avg = np.mean(kl, 1)

    with open(long_tplt.replace('run%d_eval_fastoneshot.npy', 'CS.txt'), 'w') as fw: 
    # with open(long_tplt.replace('run%d.npy', 'CS.txt'), 'w') as fw: 
    # with open(long_tplt.replace('bak', 'CS.txt'), 'w') as fw: 
        fw.write('CCS: '+'\t'.join([str(s) for s in y_eq])+'\n'+'ICS: '+'\t'.join([str(s) for s in kl_avg])+'\n')


if __name__ == '__main__':
    config, args = parse_args()
    # Ignore warnings
    warnings.simplefilter('ignore')
    main(config, args)
