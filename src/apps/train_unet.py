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
    assert cfg.run is not None, "No run specified. Add +run.task_key=foo +run.iteration=bar to program call."
    assert cfg.run.task_key is not None, "No task_key specified. Add +run.task_key=foo to program call."
    assert cfg.run.iteration is not None, "No iteration specified. Add +run.iteration=foo to program call."

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
        training='unet', 
        cfg=cfg
    )

    # get model
    task_key = cfg.run.task_key
    unet_cfg = cfg.unet[task_key]
    with open_dict(unet_cfg):
        unet_cfg.log = cfg.wandb.log
        unet_cfg.debug = cfg.debug
        unet_cfg.root = cfg.fs.root
        unet_cfg.task_key = task_key
        unet_cfg.iteration = cfg.run.iteration
    
    model = get_unet(
        unet_cfg=unet_cfg, 
        return_state_dict=False
    )

    # get trainer
    trainer = get_unet_trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        model_cfg=unet_cfg
    )
    
    trainer.fit()

if __name__ == "__main__":
    main()