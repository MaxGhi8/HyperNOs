import matplotlib.pyplot as plt
import numpy as np


def plot_hh(V, m, h, n, V_NN, m_NN, h_NN, n_NN, I_app, t, indx):
    """
    Multiple plots for Hodking-Huxley

    Args:
    V,m,h,n             => solutions given by odes
    V_NN,m_NN,h_NN,n_NN => prediction given by FNN
    I_app               => Inputs of the network
    t                   => linspace where the solutions is evaluated
    indx                => which examples to plot
    """
    n_examples = len(indx)

    I_app = I_app[indx, :]

    I_app_max = np.max(I_app) * 1.1

    V_exact = V[indx, :]
    m_exact = m[indx, :]
    h_exact = h[indx, :]
    n_exact = n[indx, :]

    V_pred = V_NN[indx, :]
    m_pred = m_NN[indx, :]
    h_pred = h_NN[indx, :]
    n_pred = n_NN[indx, :]

    V_pred_max = np.max(V_pred) * 1.2
    V_pred_min = np.min(V_pred) * 1.2

    m_pred_max = np.max(m_pred) * 1.2
    m_pred_min = np.min(m_pred) * 0.8

    h_pred_max = np.max(h_pred) * 1.2
    h_pred_min = np.min(h_pred) * 0.8

    n_pred_max = np.max(n_pred) * 1.2
    n_pred_min = np.min(n_pred) * 0.8

    fig, axs = plt.subplots(5, n_examples)
    fig.set_figheight(15)
    fig.set_figwidth(20)

    for i in range(n_examples):
        axs[0, i].plot(t, I_app[i, :])
        axs[0, i].set_title(r"$I_{{app}} = {i:.2f} $".format(i=I_app[i, 0]))
        axs[0, i].axes.set_xticklabels([])
        axs[0, i].set_ylim([0, I_app_max])
        axs[0, i].grid()

        axs[1, i].plot(t, V_exact[i, :], "b", linewidth=1.2, label="Ode15s")
        axs[1, i].plot(t, V_pred[i, :], "r--", linewidth=1.2, label="FNO")
        axs[1, i].axes.set_xticklabels([])
        axs[1, i].set_ylim([V_pred_min, V_pred_max])
        axs[1, i].grid()

        axs[2, i].plot(t, m_exact[i, :], "b", linewidth=1.2)
        axs[2, i].plot(t, m_pred[i, :], "r--", linewidth=1.2)
        # axs[2, i].set_xlabel('Time (ms)')
        axs[2, i].axes.set_xticklabels([])
        axs[2, i].set_ylim([m_pred_min, m_pred_max])
        axs[2, i].grid()

        axs[3, i].plot(t, h_exact[i, :], "b", linewidth=1.2)
        axs[3, i].plot(t, h_pred[i, :], "r--", linewidth=1.2)
        # axs[2, i].set_xlabel('Time (ms)')
        axs[3, i].axes.set_xticklabels([])
        axs[3, i].set_ylim([h_pred_min, h_pred_max])
        axs[3, i].grid()

        axs[4, i].plot(t, n_exact[i, :], "b", linewidth=1.2)
        axs[4, i].plot(t, n_pred[i, :], "r--", linewidth=1.2)
        axs[4, i].set_xlabel("Time (ms)")
        axs[4, i].set_ylim([n_pred_min, n_pred_max])
        axs[4, i].grid()

        if i == 0:  # only on the first row we want the names of the variables
            axs[0, i].set_ylabel("Current applied")
            axs[1, i].set_ylabel("V (mV)")
            axs[2, i].set_ylabel("m")
            axs[3, i].set_ylabel("h")
            axs[4, i].set_ylabel("n")
            fig.legend(loc="lower right", ncols=2)
        else:
            axs[0, i].set_yticklabels([])
            axs[1, i].set_yticklabels([])
            axs[2, i].set_yticklabels([])
            axs[3, i].set_yticklabels([])
            axs[4, i].set_yticklabels([])

    indx_str = "".join(f"{i}, " for i in indx)
    plt.suptitle(f"Test indexes number: {indx_str[:-2]}")
    plt.show()


def plot_error_hh(V, m, h, n, V_NN, m_NN, h_NN, n_NN, I_app, t, indx):
    """
    Multiple plots for Hodking-Huxley

    Args:
    V,m,h,n             => solutions given by odes
    V_NN,m_NN,h_NN,n_NN => prediction given by FNN
    I_app               => Inputs of the network
    t                   => linspace where the solutions is evaluated
    indx                => which examples to plot
    """
    n_examples = len(indx)

    I_app = I_app[indx, :]

    I_app_max = np.max(I_app) * 1.1

    V_exact = V[indx, :]
    m_exact = m[indx, :]
    h_exact = h[indx, :]
    n_exact = n[indx, :]

    V_pred = V_NN[indx, :]
    m_pred = m_NN[indx, :]
    h_pred = h_NN[indx, :]
    n_pred = n_NN[indx, :]

    V_pred_max = np.max(V_pred) * 1.2
    V_pred_min = np.min(V_pred) * 1.2

    m_pred_max = np.max(m_pred) * 1.2
    m_pred_min = np.min(m_pred) * 0.8

    h_pred_max = np.max(h_pred) * 1.2
    h_pred_min = np.min(h_pred) * 0.8

    n_pred_max = np.max(n_pred) * 1.2
    n_pred_min = np.min(n_pred) * 0.8

    err_V = np.abs(V_exact - V_pred)
    err_m = np.abs(m_exact - m_pred)
    err_h = np.abs(h_exact - h_pred)
    err_n = np.abs(n_exact - n_pred)

    err_V_max = np.max(err_V) * 1.2
    err_V_min = np.min(err_V) * 0.8
    err_m_max = np.max(err_m) * 1.2
    err_m_min = np.min(err_m) * 0.8
    err_h_max = np.max(err_h) * 1.2
    err_h_min = np.min(err_h) * 0.8
    err_n_max = np.max(err_n) * 1.2
    err_n_min = np.min(err_n) * 0.8

    fig, axs = plt.subplots(9, n_examples)
    fig.set_figheight(15)
    fig.set_figwidth(20)

    for i in range(n_examples):
        axs[0, i].plot(t, I_app[i, :])
        axs[0, i].set_title(r"$I_{{app}} = {i:.2f} $".format(i=I_app[i, 0]))
        axs[0, i].axes.set_xticklabels([])
        axs[0, i].set_ylim([0, I_app_max])
        axs[0, i].grid()

        axs[1, i].plot(t, V_exact[i, :], "b", linewidth=1.2, label="Ode15s")
        axs[1, i].plot(t, V_pred[i, :], "r--", linewidth=1.2, label="FNO")
        axs[1, i].axes.set_xticklabels([])
        axs[1, i].set_ylim([V_pred_min, V_pred_max])
        axs[1, i].grid()

        axs[2, i].semilogy(t, err_V[i, :], "b", linewidth=0.9)
        axs[2, i].axes.set_xticklabels([])
        axs[2, i].set_ylim([err_V_min, err_V_max])
        axs[2, i].grid()

        axs[3, i].plot(t, m_exact[i, :], "b", linewidth=1.2)
        axs[3, i].plot(t, m_pred[i, :], "r--", linewidth=1.2)
        axs[3, i].axes.set_xticklabels([])
        axs[3, i].set_ylim([m_pred_min, m_pred_max])
        axs[3, i].grid()
        axs[4, i].semilogy(t, err_m[i, :], "b", linewidth=0.9)

        axs[4, i].axes.set_xticklabels([])
        axs[4, i].set_ylim([err_m_min, err_m_max])
        axs[4, i].grid()
        axs[5, i].plot(t, h_exact[i, :], "b", linewidth=1.2)
        axs[5, i].plot(t, h_pred[i, :], "r--", linewidth=1.2)

        axs[5, i].axes.set_xticklabels([])
        axs[5, i].set_ylim([h_pred_min, h_pred_max])
        axs[5, i].grid()

        axs[6, i].semilogy(t, err_h[i, :], "b", linewidth=0.9)
        axs[6, i].set_ylim([err_h_min, err_h_max])
        axs[6, i].axes.set_xticklabels([])
        axs[6, i].grid()

        axs[7, i].plot(t, n_exact[i, :], "b", linewidth=1.2)
        axs[7, i].plot(t, n_pred[i, :], "r--", linewidth=1.2)
        axs[7, i].set_ylim([n_pred_min, n_pred_max])
        axs[7, i].axes.set_xticklabels([])
        axs[7, i].grid()

        axs[8, i].semilogy(t, err_n[i, :], "b", linewidth=0.9)
        axs[8, i].set_ylim([err_n_min, err_n_max])
        axs[8, i].set_xlabel("Time (ms)")
        axs[8, i].grid()

        if i == 0:  # only on the first row we want the names of the variables
            axs[0, i].set_ylabel("Current applied")
            axs[1, i].set_ylabel("V (mV)")
            axs[2, i].set_ylabel(r"$|V - V_{{\Theta}}|$")
            axs[3, i].set_ylabel("m")
            axs[4, i].set_ylabel(r"$|m - m_{{\Theta}}|$")
            axs[5, i].set_ylabel("h")
            axs[6, i].set_ylabel(r"$|h - h_{{\Theta}}$|")
            axs[7, i].set_ylabel("n")
            axs[8, i].set_ylabel(r"$|n - n_{{\Theta}}|$")
            fig.legend(loc="lower right", ncols=2)
        else:
            axs[0, i].set_yticklabels([])
            axs[1, i].set_yticklabels([])
            axs[2, i].set_yticklabels([])
            axs[3, i].set_yticklabels([])
            axs[4, i].set_yticklabels([])
            axs[5, i].set_yticklabels([])
            axs[6, i].set_yticklabels([])
            axs[7, i].set_yticklabels([])
            axs[8, i].set_yticklabels([])

    indx_str = "".join(f"{i}, " for i in indx)
    plt.suptitle(f"Test indexes number: {indx_str[:-2]}")
    plt.show()
