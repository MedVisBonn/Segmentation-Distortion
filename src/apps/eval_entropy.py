import sys, os
import hydra
from omegaconf import OmegaConf
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append('../')
from data_utils import get_eval_data
from model.unet import get_unet
from eval.auroc import get_auroc_output
from eval.eaurc import get_eaurc_output
from eval.precision_recall import get_precision_recall



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
    AUROC = False
    EAURC = False
    PR    = True

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


    # TODO:
    # eval_auroc_entropy
    # eval_eaurc_entropy

    # device handling
    device = ['cuda:0', 'cuda:0']
    if (not cfg.eval.data.subset.apply) and (cfg.run.data_key == 'brain'):
        device = ['cuda:0', 'cpu']
    unet.to(device[0])

    dfs_auroc = []
    dfs_eaurc = []
    dfs_pr = []

    if cfg.run.data_key == 'brain' and cfg.eval.data.subset.apply:
        val_key = 'val_subset'
    else:
        val_key = 'val'
    unet_name = cfg.unet[cfg.run.data_key].pre
    for key in data:
        # TODO:
        # eval_auroc_entropy
        if key != val_key and AUROC:
            auroc, fpr, tpr = get_auroc_output(
                model=unet,
                iid_data=data[val_key],
                ood_data=data[key],
                net_out=cfg.run.data_key,
                dae=False,
                umap='top2diff',
                device='cuda:0'
            )

            dfs_auroc.append(
                pd.DataFrame(
                    {
                        'auroc': auroc,
                        'fpr': fpr,
                        'tpr': tpr,
                        'data_key': cfg.run.data_key,
                        'run': cfg.run.iteration,
                        'domain': key,
                        'method': f'top2diff',
                        'unet': unet_name
                    }
                )
            )
            df = pd.concat(dfs_auroc)
            save_dir = f'{cfg.fs.repo_root}results-tmp/dae-data/'
            save_name = f'auroc_' + \
                f'{cfg.run.data_key}_' + \
                f'top2diff_' + \
                f'{cfg.unet[cfg.run.data_key].pre}_' + \
                f'{cfg.run.iteration}.csv'
            
            if cfg.debug:
                save_name = 'debug_' + save_name
            if(not os.path.exists(save_dir)):
                os.makedirs(save_dir)
            save_name = save_name.replace('__', '_')
            df.to_csv(save_dir + save_name)

        # eval_eaurc_entropy
        if EAURC:
            eaurc, aurc, risks, weights, selective_risks = get_eaurc_output(
                model=unet,
                data=data[key],
                net_out=cfg.run.data_key,
                dae=False,
                umap='top2diff',
                device='cuda:0'
            )

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
                        'domain': key,
                        'method': f'top2diff',
                        'unet': unet_name
                    }
                )
            )
            df = pd.concat(dfs_eaurc)
            save_dir = f'{cfg.fs.repo_root}results-tmp/dae-data/'
            save_name = f'eaurc_' + \
                f'{cfg.run.data_key}_' + \
                f'top2diff_' + \
                f'{cfg.unet[cfg.run.data_key].pre}_' + \
                f'{cfg.run.iteration}.csv'
            
            if cfg.debug:
                save_name = 'debug_' + save_name
            if(not os.path.exists(save_dir)):
                os.makedirs(save_dir)
            save_name = save_name.replace('__', '_')
            df.to_csv(save_dir + save_name)

        if PR:
            # Precision Recall Curve
            p_sampled, r_sampled, pr_auc = get_precision_recall(
                model=unet,
                dataset=data[key],
                net_out=cfg.run.data_key,
                dae=False,
                umap='top2diff',
                n_taus='auto',
                device=device,
            )
            
            dfs_pr.append(
                pd.DataFrame(
                    data={
                        'precision': p_sampled,
                        'recall': r_sampled,
                        'pr_auc': pr_auc,
                        'data_key': cfg.run.data_key,
                        'run': cfg.run.iteration,
                        'domain': key,
                        'method': f'top2diff',
                        'unet': unet_name
                    }
                )
            )

            df = pd.concat(dfs_pr)
            save_dir = f'{cfg.fs.repo_root}results-tmp/dae-data/'
            save_name = f'{cfg.run.data_key}_' + \
                f'pr_' + \
                f'top2diff_' + \
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

                    