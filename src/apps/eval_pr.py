import sys
import hydra
from omegaconf import OmegaConf

sys.path.append('../')
from data_utils import get_eval_data
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
    assert cfg.run.name is not None, "No name specified. Add +run.name=foo to program call."
    assert cfg.run.experiment is not None, "No experiment specified. Add +run.experiment_key=foo to program call."

    # get segmentation model
    unet, state_dict = get_unet(
        cfg=cfg, 
        return_state_dict=True
    )
    unet.load_state_dict(state_dict)

    # get data
    if cfg.run.experiment.data.subset.apply:
        subset_dict = OmegaConf.to_container(
            cfg.run.experiment.data.subset.params, 
            resolve=True, 
            throw_on_missing=True
        )
        subset_dict['unet'] = unet
    else:
        subset_dict = None

    data = get_eval_data(
        train_set=cfg.run.experiment.data.training,
        val_set=cfg.run.experiment.data.validation,
        test_sets=cfg.run.experiment.data.testing,    
        cfg=cfg,
        subset_dict=subset_dict
    )

    # get denoising models
    model, state_dict = get_daes(
        unet=unet,
        cfg=cfg, 
        return_state_dict=True
    )
    model.load_state_dict(state_dict)

    if cfg.run.experiment.data.subset.apply or cfg.run.data_key == 'heart':
        device=['cuda:0', 'cuda:0']

    else:
        # In case of calgary w/o subsetting, we need to use cpu for the precision
        # and recall value calculation. Otherwise, we run out of memory.
        device=['cuda:0', 'cpu']
        # Additionally, we filter for subsets only.
        data = {key: data[key] for key in data if 'subset' in key}


    for key in data:
        result = get_precision_recall(
            model=model,
            dataset=data[key],
            net_out=cfg.run.data_key,
            dae=True,
            umap='cross_entropy',
            device=device,
        )

        #TODO: aggregate and log results, support for other methods


