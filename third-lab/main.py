import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from tabulate import tabulate

# Required absolute tolerance for solution
EPS = 0.00001


def func(t, X):
    dX = np.zeros(X.shape)
    dX[0] = X[1]
    dX[1] = -t * X[1] - (np.power(t, 2) - 2) / np.power(t, 2) * X[0]
    return dX

def func_exact(t):
    return 1 / t

def RKF45(f, T, x0):
    rk_integ = ode(f).set_integrator("dopri5", atol=EPS).set_initial_value(x0, T[0])

    X = np.array([x0, *[rk_integ.integrate(T[i]) for i in range(1, len(T))]])

    # Split the array to values of Y and values of Y derivative
    return X[:, 0], X[:, 1]

def eulers_method(f, T, x0):
    X = np.zeros((len(T), len(x0)))
    X[0] = x0
    h = T[1] - T[0]

    for i in range(len(T) - 1):
        x_star = X[i] + h/2 * func(T[i], X[i])
        X[i + 1] = X[i] + h * f(T[i] + h/2, x_star)

    return X[:, 0]

def evaluate(h, rang):
    global T
    global Y_EXACT
    global Y_RKF45
    global Y_DER_RKF45
    global Y_RKF45_ERR
    global Y_EULER
    global Y_EULER_ERR

    x0 = np.array([1, -1])
    T = np.arange(rang[0], rang[1] + h, h)
    Y_EXACT = func_exact(T)

    Y_RKF45, Y_DER_RKF45 = RKF45(func, T, x0)
    Y_RKF45_ERR = Y_RKF45 - Y_EXACT

    Y_EULER = eulers_method(func, T, x0)
    Y_EULER_ERR = Y_EULER - Y_EXACT

def draw_graphs(values, titles, output_filename):
    fig, *ax = plt.subplots(nrows=1, ncols=len(values))

    figsizes = [0, 4, 12, 12]
    fig.set_figwidth(figsizes[len(values)])

    for i in range(len(values)):
        ax[0][i].title.set_text(titles[i])
        ax[0][i].grid()
        ax[0][i].set(xlabel='t', ylabel='y')

        if len(values[i][0]) < 25:
            ax[0][i].plot(values[i][0], values[i][1], marker='o')
        else:
            ax[0][i].plot(values[i][0], values[i][1], marker=',')

    fig.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)

def print_table():
    column_titles = ["t", "Exact value", "RKF45", "RKF45 err", "Euler's method", "Euler's method err"]
    table = []
    sample_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    it = 0
    for i in range(len(sample_values)):
        while not np.isclose(sample_values[i], T[it], atol=1e-05):
            it += 1
        table.append([round(T[it], 5), Y_EXACT[it], Y_RKF45[it], Y_RKF45_ERR[it], Y_EULER[it], Y_EULER_ERR[it]])

    print(tabulate(table, column_titles, 'pretty'))
    print("RKF45 global err:", np.sum(Y_RKF45_ERR))
    print("Euler global err:", np.sum(Y_EULER_ERR))

    print("\n\n")

def main():
    evaluate(0.1, [1, 2])
    draw_graphs(
            np.array(([T, Y_EXACT], [T, Y_RKF45], [T, Y_EULER])),
            ["Исходный график", "RKF45", "Метод ломаных Эйлера"],
            "plots\\functions-01"
    )
    draw_graphs(
            np.array(([T, Y_RKF45_ERR], [T, Y_EULER_ERR])),
            ["Погрешность RKF45", "Погрешность ломаных Эйлера"],
            "plots\\errors-01"
    )
    print("h = 0.1")
    print_table()

    evaluate(0.05, [1, 2])
    draw_graphs(
            np.array(([T, Y_EXACT], [T, Y_RKF45], [T, Y_EULER])),
            ["Исходный график", "RKF45", "Метод ломаных Эйлера"],
            "plots\\functions-005"
    )
    draw_graphs(
            np.array(([T, Y_RKF45_ERR], [T, Y_EULER_ERR])),
            ["Погрешность RKF45", "Погрешность ломаных Эйлера"],
            "plots\\errors-005"
    )
    print("h = 0.05")
    print_table()

    evaluate(0.025, [1, 2])
    draw_graphs(
            np.array(([T, Y_EXACT], [T, Y_RKF45], [T, Y_EULER])),
            ["Исходный график", "RKF45", "Метод ломаных Эйлера"],
            "plots\\functions-0025"
    )
    draw_graphs(
            np.array(([T, Y_RKF45_ERR], [T, Y_EULER_ERR])),
            ["Погрешность RKF45", "Погрешность ломаных Эйлера"],
            "plots\\errors-0025"
    )
    print("h = 0.025")
    print_table()

    evaluate(0.0125, [1, 2])
    draw_graphs(
            np.array(([T, Y_EXACT], [T, Y_RKF45], [T, Y_EULER])),
            ["Исходный график", "RKF45", "Метод ломаных Эйлера"],
            "plots\\functions-00125"
    )
    draw_graphs(
            np.array(([T, Y_RKF45_ERR], [T, Y_EULER_ERR])),
            ["Погрешность RKF45", "Погрешность ломаных Эйлера"],
            "plots\\errors-00125"
    )
    print("h = 0.0125")
    print_table()


if __name__ == "__main__":
    main()
