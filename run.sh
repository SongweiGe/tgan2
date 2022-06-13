conda activate chainer
module load cuda/9.0.176 cudnn/v7.6.5

python3 compute_inception_score.py ./config.yml
python3 compute_long_video_score.py ./config.yml