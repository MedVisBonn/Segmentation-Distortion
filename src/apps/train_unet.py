import sys
import wandb
import hydra
from omegaconf import OmegaConf, open_dict

sys.path.append('../')
from data_utils import get_train_loader
from model.unet import get_unet
from trainer.unet_trainer import get_unet_trainer



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
        training='unet', 
        cfg=cfg
    )

    # get model
    model = get_unet(
        cfg=cfg, 
        return_state_dict=False
    )

    # get trainer
    trainer = get_unet_trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        cfg=cfg
    )

    # print('Done.')
    trainer.fit()

if __name__ == "__main__":
    main()