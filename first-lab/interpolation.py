import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import lagrange, CubicSpline
from matplotlib import pyplot as plt
from matplotlib.ticker import *
from tabulate import tabulate

X_STEP = 0.1
X_STEP_COUNT = 20
X_START = -1
FUNC = lambda x: 1/(1 + 25 * np.power(x, 2))


#plot_values is an array of tuples, that hold x and y values of a particular plot
#plot_titles is an array with size equal to the size of plot_values
def draw_plots(plot_values, plot_titles, file_name):
    fig, *axes = plt.subplots(1, len(plot_values))
    
    for i in range(0, len(plot_values)):
        axes[0][i].title.set_text(plot_titles[i])
        axes[0][i].set(
            xlabel = "x",
            ylabel = "y"
        )
        axes[0][i].figure.set_figwidth(20)

        axes[0][i].xaxis.set_major_locator(AutoLocator())
        axes[0][i].yaxis.set_major_locator(AutoLocator())
     
        axes[0][i].grid(which="major", alpha=0.5)
        axes[0][i].grid(which="minor", alpha=0.2)
        
        if len(plot_values[i][0]) < 50:
            axes[0][i].plot(plot_values[i][0], plot_values[i][1], marker="o", color="black")
        else:
            axes[0][i].plot(plot_values[i][0], plot_values[i][1], marker=",", color="black")

    fig.savefig("plots\\" + file_name, bbox_inches="tight")
    plt.close(fig)


def print_table(columns, rows):
    table = []
    for i in range(0, len(rows[0])):
        table.append([j[i] for j in rows])
    print(tabulate(table, columns, tablefmt="pretty"))
    

def main():
    #func_x = np.linspace(X_START, X_START + X_STEP * X_STEP_COUNT, X_STEP_COUNT)
    #x_shifted = np.linspace(X_START + 0.05, X_START + 0.05 + X_STEP * (X_STEP_COUNT - 1), X_STEP_COUNT - 1)
    #x_fine_sample = np.linspace(X_START, X_START + X_STEP * X_STEP_COUNT, X_STEP_COUNT * 2)
    func_x = np.arange(X_START, X_START + X_STEP * (X_STEP_COUNT + 1), X_STEP)
    x_shifted = np.arange(X_START + 0.05, X_START + 0.05 + X_STEP * (X_STEP_COUNT), X_STEP)
    x_fine_sample = np.arange(X_START, X_START + (X_STEP) * (X_STEP_COUNT) + 0.05, X_STEP / 2)

    func_y = np.array(list(map(FUNC, func_x)))
    lagrange_interpolation = lagrange(func_x, func_y)
    spline_interpolation = CubicSpline(func_x, func_y)

    draw_plots(
        [
            (func_x, func_y),
            (func_x, lagrange_interpolation(func_x)),
            (func_x, spline_interpolation(func_x))
        ],
        [
            "Original function",
            "Lagrange interpolation",
            "Spline interpolation"
        ],
        "plots.png"
    )
    draw_plots(
        [
            (x_fine_sample, lagrange_interpolation(x_fine_sample)),
            (x_fine_sample, spline_interpolation(x_fine_sample))
        ],
        [
            "Lagrange interpolation",
            "Spline interpolation"
        ],
        "plots_fine.png"
    )
    draw_plots(
        [
            (x_shifted, np.array(list(map(FUNC, x_shifted)))),
            (x_shifted, lagrange_interpolation(x_shifted)),
            (x_shifted, spline_interpolation(x_shifted))
        ],
        [
            "Original function",
            "Lagrange interpolation",
            "Spline interpolation"
        ],
        "plots_shifted.png"
    )

    y_fine = np.array(list(map(FUNC, x_fine_sample)))
    draw_plots(
        [
            (x_fine_sample, lagrange_interpolation(x_fine_sample) - y_fine),
            (x_fine_sample, spline_interpolation(x_fine_sample) - y_fine)
        ],
        [
            "Lagrange error",
            "Spine error"
        ],
        "Approximation error"
    )
    draw_plots(
        [
            (x_fine_sample[8:-8], lagrange_interpolation(x_fine_sample)[8:-8] - y_fine[8:-8]),
            (x_fine_sample[8:-8], spline_interpolation(x_fine_sample)[8:-8] - y_fine[8:-8],)
        ],
        [
            "Lagrange error",
            "Spine error"
        ],
        "Approximation error 2"
    )

    print_table(
        ["x", "original", "lagrange", "spline"],
        [
            [round(x, 2) for x in x_shifted],
            np.array(list(map(FUNC, x_shifted))),
            lagrange_interpolation(x_shifted),
            spline_interpolation(x_shifted)
        ]
    )

    

if __name__ == "__main__":
    main()
