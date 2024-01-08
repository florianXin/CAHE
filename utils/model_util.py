from model.denoising_model import DM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_feat, get_cond_mode


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def create_model_and_diffusion(args, data):
    model = DM(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def get_model_args(args, data):
    # default args
    cond_feat = get_cond_feat(args)
    if  cond_feat == 'image':
        nfeats = 2
    elif cond_feat == 'fixhead':
        nfeats = 10
    
    cond_mode = get_cond_mode(args)

    return {'modeltype': '', 'nfeats': nfeats, 'translation': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 
            'num_heads': 4, 'dropout': 0.1, 'activation': "gelu", 'cond_feat': cond_feat, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'arch': args.arch, 'dataset': args.dataset}