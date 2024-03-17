import sys, os
import hydra
from omegaconf import OmegaConf
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append('../')
from data_utils import get_eval_data
from model.unet import get_unet
from model.mahalanobis_adatper import (
    get_pooling_mahalanobis_detector, 
    get_batchnorm_mahalanobis_detector
)
from eval.auroc import get_auroc_mahalanobis
from eval.eaurc import get_eaurc_mahalanobis


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
    assert cfg.eval is not None, "No eval specified. Add +eval_key=foo to program call."

    # GLOBALS (temporary)
    # AUROC = False


    # get segmentation model
    unet, state_dict = get_unet(
        cfg=cfg, 
        update_cfg_with_swivels=False,
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

    # get mahalanobis wrapper
    iid_data = DataLoader(
        data['train'], 
        batch_size=32, 
        shuffle=False, 
        drop_last=False,
        num_workers=4
    )
    pooling_mahalanobis_detector = get_pooling_mahalanobis_detector(
        swivels=['up3.0.conv_path.0.bn'] if cfg.unet[cfg.run.data_key].arch == 'default' else None,
        unet=unet,
        ledoitWolf=False,
        fit='raw',
        iid_data=iid_data,
        device='cuda:0'
    )

    batchnorm_mahalanobis_detector = get_batchnorm_mahalanobis_detector(
        swivels=['up3.0.conv_path.0.bn'] if cfg.unet[cfg.run.data_key].arch == 'default' else None,
        unet=unet,
        device='cuda:0'
    )

    dfs_auroc = []
    dfs_eaurc = []
    if cfg.run.data_key == 'brain' and cfg.eval.data.subset.apply:
        val_key = 'val_subset'
    else:
        val_key = 'val'
    for data_key in data:
        if key != val_key and key != 'train':
            ret = get_auroc_mahalanobis(
                wrapper=pooling_mahalanobis_detector,
                iid_data=data[val_key],
                ood_data=data[data_key],
                device='cuda:0'
            )
            for key in ret:
                auroc, fpr, tpr = ret[key]
                dfs_auroc.append(
                    pd.DataFrame(
                        {
                            'auroc': auroc,
                            'fpr': fpr,
                            'tpr': tpr,
                            'data_key': cfg.run.data_key,
                            'run': cfg.run.iteration,
                            'domain': data_key,
                            'method': f'Pooling_Mahalanobis_Raw_ALL',
                            'unet': cfg.unet[cfg.run.data_key].pre,
                        }
                    )
                )

            ret = get_auroc_mahalanobis(
                wrapper=batchnorm_mahalanobis_detector,
                iid_data=data[val_key],
                ood_data=data[data_key],
                device='cuda:0'
            )
            for key in ret:
                auroc, fpr, tpr = ret[key]
                dfs_auroc.append(
                    pd.DataFrame(
                        {
                            'auroc': auroc,
                            'fpr': fpr,
                            'tpr': tpr,
                            'data_key': cfg.run.data_key,
                            'run': cfg.run.iteration,
                            'domain': data_key,
                            'method': f'BatchNorm_Mahalanobis_ALL',
                            'unet': cfg.unet[cfg.run.data_key].pre,
                        }
                    )
                )
            df = pd.concat(dfs_auroc)
            save_dir = f'{cfg.fs.repo_root}results-tmp/dae-data/'
            save_name = f'auroc_{cfg.run.data_key}_' + \
                f'mahalanobis_' + \
                f'{cfg.unet[cfg.run.data_key].pre}_' + \
                f'{cfg.run.iteration}.csv'
            
            if cfg.debug:
                save_name = 'debug_' + save_name
            if(not os.path.exists(save_dir)):
                os.makedirs(save_dir)
            save_name = save_name.replace('__', '_')
            df.to_csv(save_dir + save_name)

        ret = get_eaurc_mahalanobis(
            wrapper=pooling_mahalanobis_detector,
            data=data[data_key],
            net_out=cfg.run.data_key,
            device='cuda:0'
        )
        for key in ret:
            eaurc, aurc, risks, weights, selective_risks = ret[key]
            dfs_eaurc.append(
                pd.DataFrame(
                    {
                        'eaurc': eaurc,
                        'aurc': aurc,
                        'risks': risks,
                        'weights': weights,
                        'selective_risks': selective_risks,
                        'data_key': cfg.run.data_key,
                        'run': cfg.run.iteration,
                        'domain': data_key,
                        'method': f'Pooling_Mahalanobis_Raw_ALL',
                        'unet': cfg.unet[cfg.run.data_key].pre,
                    }
                )
            )

        ret = get_eaurc_mahalanobis(
            wrapper=batchnorm_mahalanobis_detector,
            data=data[data_key],
            net_out=cfg.run.data_key,
            device='cuda:0'
        )
        for key in ret:
            eaurc, aurc, risks, weights, selective_risks = ret[key]
            dfs_eaurc.append(
                pd.DataFrame(
                    {
                        'eaurc': eaurc,
                        'aurc': aurc,
                        'risks': risks,
                        'weights': weights,
                        'selective_risks': selective_risks,
                        'data_key': cfg.run.data_key,
                        'run': cfg.run.iteration,
                        'domain': data_key,
                        'method': f'BatchNorm_Mahalanobis_ALL',
                        'unet': cfg.unet[cfg.run.data_key].pre,
                    }
                )
            )
        df = pd.concat(dfs_eaurc)
        save_dir = f'{cfg.fs.repo_root}results-tmp/dae-data/'
        save_name = f'eaurc_{cfg.run.data_key}_' + \
            f'mahalanobis_' + \
            f'{cfg.unet[cfg.run.data_key].pre}_' + \
            f'{cfg.run.iteration}.csv'
        
        if cfg.debug:
            save_name = 'debug_' + save_name
        if(not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        save_name = save_name.replace('__', '_')
        df.to_csv(save_dir + save_name)


if __name__ == "__main__":
    main()

                    