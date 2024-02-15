import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List



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
    for data in ['brain', 'heart']:
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