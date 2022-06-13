#!/usr/bin/env python3

import argparse
import warnings

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
    # xs = np.load('/fs/vulcan-projects/contrastive_learning_songweig/TATS/results/numpy_files/ucf101/ucf101_real_train.npy')
    # xs = np.load('/fs/vulcan-projects/contrastive_learning_songweig/TATS/results/numpy_files/ucf101/ucf101_real_test.npy')
    # xs = np.load('/fs/vulcan-projects/contrastive_learning_songweig/TATS/results/numpy_files/ucf101/cond_gpt_ucf_128_488_29999_epoch=6-step=459999-train_topp0.92_topk2048_eval.npy')
    # xs = [np.load('/fs/vulcan-projects/contrastive_learning_songweig/TATS/results/numpy_files/ucf101/cond_gpt_ucf_128_488_29999_epoch=21-step=1349999-train_topp0.80_topk2048_run%d_eval.npy'%i) for i in [0, 1, 2, 4, 5]]
    # xs = [np.load('/fs/vulcan-projects/contrastive_learning_songweig/TATS/results/numpy_files/ucf101/cond_gpt_ucf_128_488_29999_epoch=21-step=1349999-train_topp0.80_topk2048_run%d_eval.npy'%i) for i in [3, 4, 5, 6, 7]]
    xs = [np.load('/fs/vulcan-projects/contrastive_learning_songweig/TATS/results/numpy_files/ucf101/uncond_gpt_ucf_128_488_29999_epoch=21-step=1349999-train_topp0.92_topk16384_run%d_eval.npy'%i) for i in [7, 2, 5, 6, 8]]
    xs = np.concatenate(xs, 0)
    xs = np.transpose(xs.astype(np.float32), (0, 4, 1, 2, 3))/255.*2-1
    scores = []
    # for i in range(args.n_loops):
        # print('Loop {}'.format(i))
        # xs = tgan2.evaluations.inception_score.make_samples(
        #     gen, batchsize=args.batchsize,
        #     n_samples=args.n_samples, n_frames=args.n_frames)
    mean, std = tgan2.evaluations.inception_score.inception_score(
        classifier, xs, args.batchsize, splits=1)
    print(f'{mean} +- {std}')
    scores.append(chainer.backends.cuda.to_cpu(mean))

    scores = numpy.asarray(scores)
    mean, std = float(numpy.mean(scores)), float(numpy.std(scores))
    print(mean, std)

    result = {'mean': mean, 'std': std}
    open(args.out, 'w').write(yaml.dump(result, default_flow_style=False))



if __name__ == '__main__':
    config, args = parse_args()
    # Ignore warnings
    warnings.simplefilter('ignore')
    main(config, args)
