import sys
import wandb
import hydra
from omegaconf import OmegaConf

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
    assert cfg.dae.name is not None, "No name specified. Add +run.name=foo to program call."

    # set up wandb
    if cfg.wandb.log:
        run = wandb.init(
            reinit=True, 
            name=cfg.wandb.name if cfg.wandb.name is not None else None,
            project=cfg.wandb.project,
            config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )

    # get data
    train_loader, val_loader = get_train_loader(
        training='dae', 
        cfg=cfg
    )

    # get unet
    unet, state_dict = get_unet(
        cfg=cfg, 
        update_cfg_with_swivels=True,
        return_state_dict=True,
    )
    unet.load_state_dict(state_dict)
    
    # get dae models
    model = get_daes(
        unet=unet,
        cfg=cfg,
        return_state_dict=False
    )

    # get trainer
    trainer = get_dae_trainer(
        cfg=cfg,
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader
    )

    # start optimization
    trainer.fit()

    # update W&B config with params that got added during training
    if cfg.wandb.log:
        run.config.update(
            OmegaConf.to_container(
                cfg, 
                resolve=True, 
                throw_on_missing=True
            )
        )


if __name__ == "__main__":
    main()