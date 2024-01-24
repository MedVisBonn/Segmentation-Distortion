import sys
import wandb
import hydra
from omegaconf import OmegaConf, open_dict

sys.path.append('../')
from data_utils import get_train_loader
from model.unet import get_unet
from model.dae import get_daes
from trainer.ae_trainerV2 import get_dae_trainer



@hydra.main(
    config_path='../configs', 
    config_name='basic_config.yaml',
    version_base=None
)
def main(
    cfg
):
    # Check if run is in cfg (has to be specified on program call)
    assert cfg.run is not None, "No run specified. Add +run.task_key=foo +run.iteration=bar to program call."
    assert cfg.run.task_key is not None, "No task_key specified. Add +run.task_key=foo to program call."
    assert cfg.run.iteration is not None, "No iteration specified. Add +run.iteration=foo to program call."
    assert cfg.run.name is not None, "No name specified. Add +run.name=foo to program call."
    task_key = cfg.run.task_key

    # set up wandb
    if cfg.wandb.log:
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        run = wandb.init(
            reinit=True, 
            name=cfg.wandb.name if cfg.wandb.name is not None else None,
            project=cfg.wandb.project, 
        )

    # get data
    train_loader, val_loader = get_train_loader(
        training='dae', 
        cfg=cfg
    )

    # get segmentation model
    unet_cfg = cfg.model.unet[task_key]
    with open_dict(unet_cfg):
        unet_cfg.iteration = cfg.run.iteration
        unet_cfg.root = cfg.fs.root
    unet, state_dict = get_unet(
        unet_cfg=unet_cfg, 
        return_state_dict=True
    )
    unet.load_state_dict(state_dict)

    # get dae models
    dae_config = OmegaConf.load(f'../configs/dae_config.yaml')
    with open_dict(dae_config):
        dae_config.log = cfg.wandb.log
        dae_config.debug = cfg.debug
        dae_config.root = cfg.fs.root
        dae_config.task_key = task_key
        dae_config.iteration = cfg.run.iteration
        dae_config.name = cfg.run.name

    daes = get_daes(arch=dae_config.arch)

    # get trainer
    trainer = get_dae_trainer(
        dae_config=dae_config, 
        daes=daes, 
        model=unet, 
        train_loader=train_loader, 
        val_loader=val_loader
    )

    trainer.fit()


if __name__ == "__main__":
    main()