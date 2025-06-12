import matplotlib.pyplot as plt
import logging

def close_figures():
    """关闭所有打开的matplotlib图形，释放内存"""
    plt.close('all')
    logging.debug("All matplotlib figures closed.")

def setup_matplotlib_style():
    """设置matplotlib的默认样式"""
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14 