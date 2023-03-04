import numpy as np
from scipy.linalg import lu, solve, norm


# n - размер матрицы
# Гарантировано генерирует инвертированную матрицу
def generate_random_matrix(n):
    m = np.random.rand(n, n)
    mx = np.sum(np.abs(m), axis=1)
    np.fill_diagonal(m, mx)

    return m


def get_inverse(A: np.matrix):
    A_inv = np.zeros(A.shape)
    E = np.identity(A.shape[0])
    P, L, U = lu(A)

    for i in range(0, A.shape[0]):
        Ei = E[:, i]

        Zi = solve(L, Ei)
        X = solve(U, Zi)
        A_inv[:, i] = X.flatten()

    return A_inv


def get_R_matrix(A: np.matrix):
    A_inv = get_inverse(A)
    E = np.identity(A_inv.shape[0])
    return np.subtract(np.matmul(A_inv, A), E)


def get_norm(A: np.matrix):
    return np.max([np.sum(np.abs(k)) for k in A], axis=None)


def get_cond(A: np.matrix):
    return get_norm(A) * get_norm(get_inverse(A))


def construct(eps):
    x_k = lambda k: 1 + np.cos(k) / np.power(np.sin(k), 2)
    x_eps = lambda eps: 1 + np.cos(1) / np.power(np.sin(1 + eps), 2)

    A = np.matrix([[np.power(x_k(k), N) for k in range(1, 5)] for N in range(0, 4)])

    new_row = [np.power(x_k(k), 4) for k in range(1, 5)]
    new_column = [[np.power(x_eps(eps), N)] for N in range(0, 5)]

    A = np.vstack((A, new_row))
    A = np.hstack((A, new_column))

    return A


def main():
    eps_arr = [0.001, 0.00001, 0.000001]
    A_arr = [construct(eps) for eps in eps_arr]

    np.set_printoptions(linewidth=150)

    for i in range(0, 3):
        print("Epsilon:", eps_arr[i])
        print("\t\t\t\t   Matrix:\n", A_arr[i], "\n")
        print("\t\t\t\t\tR:\n", get_R_matrix(A_arr[i]), "\n")
        print("Norm:", get_norm(A_arr[i]))
        print("Conditionality:", get_cond(A_arr[i]))
        print("\n", "--" * 40, "\n")


if __name__ == "__main__":
    main()
