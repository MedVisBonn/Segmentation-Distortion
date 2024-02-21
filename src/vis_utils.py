import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List



def plot_pr_eval(
    data_key: str = 'brain',
    # data_key: str = 'heart',

    architectures: List[str] = [
        'ResDAE-8',
        'ResDAE-32',
        'ResDAE-64',
    ],
    unets: List[str] = [
        # 'default-8', 
        # 'default-16', 
        'monai-8-4-4',
        # 'monai-16-4-4',
        # 'monai-16-4-8',
        # 'monai-32-4-4',
        # 'monai-64-4-4',
        # 'swinunetr'
    ],
):
    dfs = []
    for arch in architectures:
        for unet in unets:
            dfs.append(pd.read_csv(
                f'../../results-tmp/dae-data/{data_key}_{arch}_{unet}_0.csv'
            ))
    df = pd.concat(dfs)

    if len(unets) == 1:
        df['method_with_auc'] = df.apply(lambda row: f"{row['method']} ({row['pr_auc']:.3f})", axis=1)

    # Creating the figure and axes for the subplots
    n = df['domain'].nunique()
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

    for i, domain in enumerate(df['domain'].unique()):
        
        sns.lineplot(
            data=df[df['domain'] == domain], 
            x='recall',
            y='precision', 
            hue='method_with_auc' if len(unets) == 1 else 'method',
            style='unet',
            ax=axes[i]
        )
    return fig



def plot_unet_eval(
    data_keys: List[str] = [
        'brain', 
        'heart'
    ],
    architectures: List[str] = [
        'default-8', 
        'default-16', 
        'monai-16-4-4'
    ],
    data_dir: str = '../results/unet'
):
    dfs = []
    for data in data_keys:
        for arch in architectures:
            dfs.append(pd.read_csv(f'{data_dir}/df_{data}_{arch}_0.csv'))

    df = pd.concat(dfs)

    # Creating the figure and axes for the subplots
    n = len(data_keys)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

    # Plot for each dataset
    for i, key in enumerate(data_keys):
        sns.lineplot(
            data=df[df['Data'] == key], 
            x='Domain', 
            y='value', 
            hue='Model', 
            style='variable', 
            markers=True, 
            dashes=False, 
            markersize=10, 
            marker='x', 
            ax=axes[i]
        )
        axes[i].set_title(f'{key.capitalize()} Data')

    plt.tight_layout()

    return fig