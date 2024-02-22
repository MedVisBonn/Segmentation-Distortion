import sys
import hydra
from omegaconf import OmegaConf
import pandas as pd

sys.path.append('../')
from data_utils import get_eval_data
from utils import load_state_dict_for_modulelists
from model.unet import get_unet
from model.dae import get_daes
from eval.pixel_wise import get_precision_recall



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
    assert cfg.dae.name is not None, "No name specified. Add +dae.name=foo to program call."
    assert cfg.eval is not None, "No eval specified. Add +eval_key=foo to program call."

    # print(OmegaConf.to_yaml(cfg))
    
    # get segmentation model
    unet, state_dict = get_unet(
        cfg=cfg, 
        update_cfg_with_swivels=True,
        return_state_dict=True
    )
    unet.load_state_dict(state_dict)

    # get data
    if cfg.eval.data.subset.apply:
        subset_dict = OmegaConf.to_container(
            cfg.eval.data.subset.params, 
            resolve=True, 
            throw_on_missing=True
        )
        subset_dict['unet'] = unet
    else:
        subset_dict = None

    data = get_eval_data(
        train_set=cfg.eval.data.training,
        val_set=cfg.eval.data.validation,
        test_sets=cfg.eval.data.testing,    
        cfg=cfg,
        subset_dict=subset_dict
    )

    print('N cases per domain:')
    for key in data:
        print(f'    {key}: {len(data[key])}')

    # get denoising models
    model, state_dict = get_daes(
        unet=unet,
        cfg=cfg, 
        return_state_dict=True
    )

    if cfg.dae.placement == 'all':
        model = load_state_dict_for_modulelists(model, state_dict)
    elif cfg.dae.placement == 'bottleneck':
        model.load_state_dict(state_dict)
    # model.load_state_dict(state_dict)

    # In case of calgary w/o subsetting, we need to use cpu for the precision
    # and recall value calculation. In all other cases, calculations are done on GPU.
    device = ['cuda:0', 'cuda:0']
    if (not cfg.eval.data.subset.apply) and (cfg.run.data_key == 'brain'):
        device = ['cuda:0', 'cpu']


    dfs = []
    for key in data:
        p_sampled, r_sampled, pr_auc = get_precision_recall(
            model=model,
            dataset=data[key],
            net_out=cfg.run.data_key,
            dae=True,
            umap='cross_entropy',
            device=device,
        )

        dfs.append(
            pd.DataFrame(
                data={
                    'precision': p_sampled,
                    'recall': r_sampled,
                    'pr_auc': pr_auc,
                    'data_key': cfg.run.data_key,
                    'run': cfg.run.iteration,
                    'domain': key,
                    'method': cfg.dae.name,
                    'unet': cfg.unet[cfg.run.data_key].pre,
                }
            )
        )

        df = pd.concat(dfs)

        save_name = f'{cfg.fs.root}results-tmp/dae-data/' + \
            f'{cfg.run.data_key}_' + \
            f'{cfg.dae.name}_' + \
            f'{cfg.unet[cfg.run.data_key].pre}_' + \
            f'{cfg.run.iteration}.csv'
        df.to_csv(save_name)

if __name__ == "__main__":
    main()

