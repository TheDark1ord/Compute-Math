import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tabulate import tabulate
import PyQt5

INTEGRAL_FUNC_1 = lambda x : np.power(np.abs(1 - np.power(x, 2)), -1)
INTEGRAL_FUNC_2 = lambda x : np.power(np.abs(1 - np.power(x, 2)), -0.5)

START = 0
END = 2.14


def draw_func():
    func_x = np.arange(START + 0.001, END + 0.01, 0.01)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(10)

    ax1.set_ylim([0, 10])
    ax2.set_ylim([0, 10])
    ax1.grid()
    ax2.grid()

    ax1.title.set_text("m = -1")
    ax2.title.set_text("m = -0.5")
    ax1.set(xlabel="x", ylabel="y")
    ax2.set(xlabel="x", ylabel="y")

    ax1.plot(func_x, np.array(list(map(INTEGRAL_FUNC_1, func_x))))
    ax2.plot(func_x, np.array(list(map(INTEGRAL_FUNC_2, func_x))))

    fig.savefig("plots\\Integral function", bbox_inches="tight")
    plt.close(fig)

def print_table(columns, rows):
    table = []
    for i in range(0, len(rows[0])):
        table.append([j[i] for j in rows])
    print(tabulate(table, columns, tablefmt="pretty"))

def get_integral_value(func, eps):
    return quad(func, START, 1 - eps, limit=30)[0] +\
            quad(func, 1 + eps, END, limit=30)[0]

def main():
    draw_func()

    eps_val = []
    integ_val_1 = []
    integ_val_2 = []

    eps = 0.1
    for i in range(0, 20):
        integ_val_1.append(get_integral_value(INTEGRAL_FUNC_1, eps))
        integ_val_2.append(get_integral_value(INTEGRAL_FUNC_2, eps))

        eps_val.append("1e-" + str(i+1))
        eps = eps / 10

    print_table(["eps", "integral, m=-1", "integral, m=-0.5"], (eps_val, integ_val_1, integ_val_2))

if __name__ == "__main__":
    main()
