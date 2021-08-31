from matplotlib import pyplot as plt

id_to_category = {
    '02691156': 'airplane',
    '02958343': 'car',
    '03001627': 'chair',
    '04379243': 'table',
    '03636649': 'lamp'

}


def plot_pcds(filename, pcds, titles, use_color=[], color=None, suptitle='', sizes=None, cmap='YlGn', zdir='y',
              xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 3))
    for i in range(1):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            clr = color[j]
            if clr is None or not use_color[j]:
                clr = pcd[:, 0]

            ax = fig.add_subplot(1, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            if j == 2:
                ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=clr, s=size, cmap='Reds', vmin=-1, vmax=0.5)
            else:
                ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=clr, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j], y=-0.01)
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    if filename in ["gan", "vae"]:
        filename = "/home/beecadox/Thesis/BiGAN/results/" + filename + "/chair/images/epoch_" + suptitle + ".jpg"
    else:
        plt.suptitle(id_to_category[suptitle.split("/")[0]] + "--" + suptitle.split("/")[1].split(".")[0])
        if titles.__len__() == 3:
            filename = "../data_plots/results/" + id_to_category[suptitle.split("/")[0]] + "--" + \
                       suptitle.split("/")[1].split(".")[0] + ".jpg"
        else:
            filename = "../data_plots/" + \
                       suptitle.replace(suptitle.split("/")[0], id_to_category[suptitle.split("/")[0]]).split(".")[
                           0] + ".jpg"
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
