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
    assert cfg.run is not None, "No run specified. Add +run.data_key=foo +run.iteration=bar to program call."
    assert cfg.run.data_key is not None, "No data_key specified. Add +run.data_key=foo to program call."
    assert cfg.run.iteration is not None, "No iteration specified. Add +run.iteration=foo to program call."
    assert cfg.run.name is not None, "No name specified. Add +run.name=foo to program call."
    data_key = cfg.run.data_key

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
    unet_cfg = cfg.unet[data_key]
    with open_dict(unet_cfg):
        unet_cfg.iteration = cfg.run.iteration
        unet_cfg.root = cfg.fs.root
    unet, state_dict = get_unet(
        unet_cfg=unet_cfg, 
        return_state_dict=True
    )
    unet.load_state_dict(state_dict)

    # get dae models
    daes = get_daes(
        arch=cfg.dae.arch, 
        model=cfg.dae.model,
        disabled_ids=cfg.dae.trainer.disabled_ids
    )

    # get trainer
    trainer_config = cfg.dae.trainer
    with open_dict(trainer_config):
        trainer_config.log = cfg.wandb.log
        trainer_config.debug = cfg.debug
        trainer_config.root = cfg.fs.root
        trainer_config.data_key = data_key
        trainer_config.iteration = cfg.run.iteration
        trainer_config.name = cfg.run.name

    trainer = get_dae_trainer(
        trainer_config=trainer_config, 
        daes=daes, 
        model=unet, 
        train_loader=train_loader, 
        val_loader=val_loader
    )

    trainer.fit()


if __name__ == "__main__":
    main()