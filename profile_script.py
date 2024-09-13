import cProfile
from plot_utils import PlotUtils


def main():
    PlotUtils.draw_plots()


if __name__ == '__main__':
    cProfile.run('main()')
