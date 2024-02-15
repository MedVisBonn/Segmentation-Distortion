import sys
import numpy as np
import hydra
from omegaconf import OmegaConf

sys.path.append('../')
from data_utils import get_eval_data
from model.unet import get_unet
from eval.unet_test import eval_set, get_df_from_dict


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

    # get model
    unet, state_dict = get_unet(
        cfg=cfg, 
        return_state_dict=True
    )
    unet.load_state_dict(state_dict)
    unet.cuda()

    # get data
    data = get_eval_data(
        train_set=cfg.eval.data.training,
        val_set=cfg.eval.data.validation,
        test_sets=cfg.eval.data.testing,    
        cfg=cfg
    )

    # evaluate model on each dataset in data
    metrics = {}
    print('Evaluating model on each dataset:')
    for key in data:
        print(f'    {key}: ...', end='\r')
        metrics[key] = eval_set(
            cfg=cfg,
            model=unet,
            dataset=data[key]
        )
        print(f"    {key}: \u2713    ")

    # convert to pandas dataframe, melt and save
    df = get_df_from_dict(
        cfg=cfg,
        metrics=metrics
    )
    data_key = cfg.run.data_key
    save_name = f'df_{cfg.run.data_key}_{cfg.unet[data_key].pre}_{cfg.run.iteration}.csv'

    df.to_csv(save_name, index=False)
    print('\n', df)


if __name__ == "__main__":
    main()