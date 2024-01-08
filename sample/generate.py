# This code is based on https://github.com/openai/guided-diffusion and https://github.com/GuyTevet/motion-diffusion-model 
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
import shutil
from data_loaders.tensors import collate
import codecs as cs
import pandas as pd
from tqdm import tqdm
from os.path import join as pjoin


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    n_frames = 5
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))

    if args.test_txt != '' and args.cond_path != '':
        assert os.path.exists(args.test_txt)
        with open(args.test_txt, 'r') as file:
            test_list = file.readlines()
        test_list = [test.replace('\n', '') for test in test_list]
        
    print('Loading dataset...')
    data = load_dataset(args, n_frames)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    for num in tqdm(test_list):
        if num != "":
            out_name = out_path + '_' + num

            c = np.load(pjoin(args.cond_path, num + '.npy'))
            cond = [c]
            collate_args = [{'input': torch.zeros(n_frames), 'lengths': n_frames}]
            collate_args = [dict(arg, cond=m) for arg, m in zip(collate_args, cond)]
            _, model_kwargs = collate(collate_args)

            args.batch_size = 1
            
            all_inputs = []
            all_lengths = []
            all_conds = []

            for rep_i in range(args.num_repetitions):
                print(f'### Sampling [repetitions #{rep_i}]')

                # add CFG scale to batch
                if args.guidance_param != 1:
                    model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

                sample_fn = diffusion.p_sample_loop

                sample = sample_fn(
                    model,
                    (args.batch_size, model.nfeats, 1, n_frames),  # BUG FIX
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )

                all_conds += model_kwargs['y']["cond"]

                all_inputs.append(sample.cpu().numpy())
                all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
                print(f"created {len(all_inputs) * args.batch_size} samples")

            all_inputs = np.concatenate(all_inputs, axis=0)
            all_inputs = all_inputs[:args.batch_size]  # [bs, njoints, 6, seqlen]
            all_conds = all_conds[:args.batch_size]
            all_lengths = np.concatenate(all_lengths, axis=0)[:args.batch_size]

            if os.path.exists(out_name):
                shutil.rmtree(out_name)
            os.makedirs(out_name)

            npy_path = os.path.join(out_name, 'results.npy')
            print(f"saving results file to [{npy_path}]")
            np.save(npy_path, all_inputs)

            npy_path = os.path.join(out_name, 'conds.npy')
            np.save(npy_path+'', all_conds)

            with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
                fw.write('\n'.join([str(l) for l in all_lengths]))

            abs_path = os.path.abspath(out_name)
            print(f'[Done] Results are at [{abs_path}]')


def load_dataset(args, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              split='test')
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
