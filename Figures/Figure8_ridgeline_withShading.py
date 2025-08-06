import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# # Create the data
# rs = np.random.RandomState(1979)
# x = rs.randn(120)
# g = np.tile(list("ABCD"), 30)
# h = np.tile(list("XYZ"), 40)
#
# # Generate df
# df = pd.DataFrame(dict(x = x, g = g, h = h))
#
# # Initialize the FacetGrid object
# pal = sns.cubehelix_palette(4, rot = -0.25, light = 0.7)
# g = sns.FacetGrid(df, col = 'h', hue = 'h', row = 'g', aspect = 3, height= 1, palette = pal)
#
# # Draw the densities
# g = g.map(sns.kdeplot, 'x', shade = True, alpha = 1, lw = 1, bw = 0.8)
# g = g.map(sns.kdeplot, 'x', color= 'w', lw = 1, bw = 0.8)
# g = g.map(plt.axhline, y = 0, lw = 1)
#
# for ax in g.axes.flat:
#     ax.set_title("")
#
# # Adjust title and axis labels directly
# for i in range(4):
#     g.axes[i,0].set_ylabel('L {:d}'.format(i))
# for i in range(3):
#     g.axes[0,i].set_title('Top {:d}'.format(i))
#
# # generate a gradient
# cmap = 'coolwarm'
# x = np.linspace(0,1,100)
# for ax in g.axes.flat:
#     im = ax.imshow(np.vstack([x,x]), aspect='auto', extent=[*ax.get_xlim(), *ax.get_ylim()], cmap=cmap, zorder=2)
#     path = ax.collections[0].get_paths()[0]
#     patch = matplotlib.patches.PathPatch(path, transform=ax.transData)
#     im.set_clip_path(patch)
#
# g.set_axis_labels(x_var = 'Total Amount')
# g.set(yticks = [])
# plt.show()
'''NEW PLOT'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")  # or 'Qt5Agg', 'Agg' (non-interactive), etc.

from numpy import loadtxt
import json
from matplotlib import colors
fig, axs = plt.subplots(nrows=2, figsize=(6, 4), sharex=True, sharey=True)
fig.set_facecolor("none")
# cmap = colors.ListedColormap(['white','red','blue','green'])
# bounds=[0, 0.005, 0.1, 0.2, 1]
# norm = colors.BoundaryNorm(bounds, cmap.N)
# Read the text file
with open('Figure8_data/traindata_2.json', 'r') as file:
    data = json.load(file)
with open('Figure8_data/shap_values_2_0.json', 'r') as file:
    shap_data = json.load(file)
with open('Figure8_data/shap_values_2_1.json', 'r') as file:
    shap_data1 = json.load(file)
# with open('shap_values_6_2.json', 'r') as file:
#     shap_data2 = json.load(file)

data = np.array(data)
arr_shap = np.array(shap_data)
arr_shap1 = np.array(shap_data1)
# arr_shap2 = np.array(shap_data2)

for i, ax in enumerate(axs, 1):
    x = data[:, i-1]
    # x = arr_shap[:, i - 1]
    x_norm = (x - min(x)) / (max(x) - min(x))
    # sns.kdeplot(arr_shap[:,i-1])
    # sns.kdeplot(arr_shap1[:,i-1])
    # plt.show()
    sns.kdeplot(arr_shap[:,i-1],
                fill=True, color="black", alpha=0, linewidth=2, legend=False, ax=ax)##225ea8
    im = ax.imshow(np.vstack([np.sort(x_norm), np.sort(x_norm)]),
                   cmap='Accent',#'PRGn'
                   aspect="auto",
                   extent=[*ax.get_xlim(), *ax.get_ylim()]
                   )
    path = ax.collections[0].get_paths()[0]
    patch = mpl.patches.PathPatch(path, transform=ax.transData)
    im.set_clip_path(patch)

    sns.kdeplot(arr_shap1[:,i-1],
                fill=True, color="#2ca02c", alpha=0, linewidth=2, legend=False, ax=ax)
#ffffcc do nothing, reopt 20 black, reopt 50 41b6c4, reopt5 225ea8
    im = ax.imshow(np.vstack([np.sort(x_norm), np.sort(x_norm)]),
                   cmap='Accent',
                   aspect="auto",
                   extent=[*ax.get_xlim(), *ax.get_ylim()]
                   )
    path = ax.collections[1].get_paths()[0]
    patch = mpl.patches.PathPatch(path, transform=ax.transData)
    im.set_clip_path(patch)

    # sns.kdeplot(arr_shap2[:,i-1],
    #             fill=True, color="#377eb8", alpha=0, linewidth=2, legend=False, ax=ax)
    #
    # im = ax.imshow(np.vstack([np.sort(x_norm), np.sort(x_norm)]),
    #                cmap='PRGn',
    #                aspect="auto",
    #                extent=[*ax.get_xlim(), *ax.get_ylim()]
    #                )
    # path = ax.collections[2].get_paths()[0]
    # patch = mpl.patches.PathPatch(path, transform=ax.transData)
    # im.set_clip_path(patch)

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if i != 12:
        ax.tick_params(axis="x", length=0, labelsize=20)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_facecolor("none")

#fig.subplots_adjust(hspace=-0.5)
plt.tight_layout()
plt.show()