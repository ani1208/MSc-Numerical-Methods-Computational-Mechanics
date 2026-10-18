import ml_collections

#get config params for the run
def get_base_config():

    config = ml_collections.ConfigDict()

    # weights and biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "DEEPXDE-burgers"
    wandb.folder = "cases/burgers"
    wandb.name = "baseline"
    wandb.tag = None

	# Arch 
    config.arch = arch = ml_collections.ConfigDict()
    arch.num_layers = 3
    arch.hidden_dim = 50
    arch.out_dim = 1
    arch.input_dim = 3
    arch.activation = "tanh"
    arch.fourier_enabled = False
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 2.0, "embed_dim": 128})

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer_1 = "adam"
    optim.optimizer_2 = "L-BFGS"
    optim.use_lbfgs = False
    optim.lr_schedule = "None"
    optim.learning_rate = 1e-4
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.separate_enabled = False
    optim.schedule = [
        [5000, 1e-3],
        [10000, 1e-3],
        [15000, 3e-4],
        [15000, 1e-4],
        [10000, 1e-4],
        [10000, 3e-5],
        [10000, 3e-6],
    ]

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.seed = 1
    training.max_steps_optimizer_1 = 1000
    training.max_steps_optimizer_2 = 5000
    training.sampling = "random"
    training.sampling_schedule = ml_collections.ConfigDict({
        "every": 100,  
    })
    training.anchors = False
    training.sampling_points_domain = 5000
    training.sampling_points_boundary = 4000
    training.exact_bc_enforcement = False
    training.rar_enabled = False
    training.rar_points = 50
    training.rar_itr_ = 0
    
    # Global weightings for different loss terms
    config.loss_weighting = loss_weighting = ml_collections.ConfigDict()
    loss_weighting.gradnorm = True
    loss_weighting.loss_weights = ml_collections.ConfigDict(
        {   
            "pde_burgers": 1.0,
            "bc": 1.0,
            "ic": 1.0,
        }
    )
    loss_weighting.update_schedule = ml_collections.ConfigDict({
        "start": 10,
        "every": 1000, 
    })
    loss_weighting.momentum = 0.9
    loss_weighting.clip = 100
    loss_weighting.explicit_enabled = False
    loss_weighting.explicit_scale = 50
    loss_weighting.explicit_indices = [0, 9]

    # Pseudo-time stepping for the PDE residuals
    config.pseudo_time = pseudo_time = ml_collections.ConfigDict()
    pseudo_time.enabled = False
    pseudo_time.strategy = "fixed"  
    pseudo_time.pts_tau = ml_collections.ConfigDict({"u": 0.1, "v": 0.1, "p": 0.1})
    pseudo_time.momentum = 0.9
    pseudo_time.prev_net_update_every = 1

    # Curriculum Re ramp
    config.re_ramp = re_ramp = ml_collections.ConfigDict()
    re_ramp.enabled = False
    re_ramp.schedule = [
        [0, 1e3],
        [10000, 1e4],
        [20000, 1e5],
        [30000, 3e5],
        [40000, 8e5],
        [50000, 2e6],
        [60000, 2.5e6],
    ]
    
    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 10
    logging.log_errors = True
    logging.log_lr = True
    logging.log_losses = True
    logging.log_raw_losses = False
    logging.log_loss_weights = True
    logging.log_pts_weights = True

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.ckpt_path = ml_collections.config_dict.placeholder(str)
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 2

    return config