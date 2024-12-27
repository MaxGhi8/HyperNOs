import matplotlib.pyplot as plt
import numpy as np


def plot_phase(V, w, V_NN, w_NN, I_app, t, indx):
    """
    Multiple plots for FHN

    Args:
    V,w        => solutions given by odes
    V_NN, w_NN => prediction given by FNN
    I_app      => Inputs of the network
    t          => linspace where the solutions is evaluated
    indx       => which examples to plot
    """
    n_examples = len(indx)

    I_app = I_app[indx, :]

    I_app_max = np.max(I_app) * 1.1

    V_exact = V[indx, :]
    w_exact = w[indx, :]

    V_pred = V_NN[indx, :]
    w_pred = w_NN[indx, :]

    # V_pred = predicted[0,:,:]
    # w_pred = predicted[1,:,:]

    V_pred_max = np.max(V_pred) * 1.2
    V_pred_min = np.min(V_pred) * 1.2

    w_pred_max = np.max(w_pred) * 1.2
    w_pred_min = np.min(w_pred) * 1.2

    fig, axs = plt.subplots(4, n_examples)
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

        axs[2, i].plot(t, w_exact[i, :], "b", linewidth=1.2)
        axs[2, i].plot(t, w_pred[i, :], "r--", linewidth=1.2)
        axs[2, i].set_xlabel("Time (ms)")
        axs[2, i].set_ylim([w_pred_min, w_pred_max])
        axs[2, i].grid()

        axs[3, i].plot(V_exact[i, :], w_exact[i, :], "b", linewidth=1.2)
        axs[3, i].plot(V_pred[i, :], w_pred[i, :], "r--", linewidth=1.2)
        axs[3, i].set_xlabel("w")
        axs[3, i].set_ylim([w_pred_min, w_pred_max])
        axs[3, i].set_xlim([V_pred_min, V_pred_max])
        axs[3, i].grid()

        if (
            i == 0
        ):  # just print the label, and for the first plot i set the label for the legend
            axs[0, i].set_ylabel("Current applied")
            axs[1, i].set_ylabel("V (mV)")
            axs[2, i].set_ylabel("w")
            axs[3, i].set_ylabel("V")
            fig.legend(loc="lower right", ncols=2)
        else:
            axs[0, i].set_yticklabels([])
            axs[1, i].set_yticklabels([])
            axs[2, i].set_yticklabels([])
            axs[3, i].set_yticklabels([])

    indx_str = "".join(f"{i}, " for i in indx)
    plt.suptitle(f"Test indexes number: {indx_str[:-2]}")
    plt.show()


def plot_fhn(V, w, V_NN, w_NN, I_app, t, indx):
    """
    Multiple plots for FHN with the phase space

    Args:
    V,w        => solutions given by odes
    V_NN, w_NN => prediction given by FNN
    I_app      => Inputs of the network
    t          => linspace where the solutions is evaluated
    indx       => which examples to plot
    """
    n_examples = len(indx)

    I_app = I_app[indx, :]

    I_app_max = np.max(I_app) * 1.1

    V_exact = V[indx, :]
    w_exact = w[indx, :]

    V_pred = V_NN[indx, :]
    w_pred = w_NN[indx, :]

    # V_pred = predicted[0,:,:]
    # w_pred = predicted[1,:,:]

    V_pred_max = np.max(V_pred) * 1.2
    V_pred_min = np.min(V_pred) * 1.2

    w_pred_max = np.max(w_pred) * 1.2
    w_pred_min = np.min(w_pred) * 1.2

    fig, axs = plt.subplots(3, n_examples)
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

        axs[2, i].plot(t, w_exact[i, :], "b", linewidth=1.2)
        axs[2, i].plot(t, w_pred[i, :], "r--", linewidth=1.2)
        axs[2, i].set_xlabel("Time (ms)")
        axs[2, i].set_ylim([w_pred_min, w_pred_max])
        axs[2, i].grid()

        if i == 0:  # only on the first row we want the names of the variables
            axs[0, i].set_ylabel("Current applied")
            axs[1, i].set_ylabel("V (mV)")
            axs[2, i].set_ylabel("w")
            fig.legend(loc="lower right", ncols=2)
        else:
            axs[0, i].set_yticklabels([])
            axs[1, i].set_yticklabels([])
            axs[2, i].set_yticklabels([])
    indx_str = "".join(f"{i}, " for i in indx)
    plt.suptitle(f"Test indexes number: {indx_str[:-2]}")
    plt.show()


def plot_error_fhn(V, w, V_NN, w_NN, I_app, t, indx):
    """
    Multiple plots for FHN with the point wise error

    Args:
    V,w        => solutions given by odes
    V_NN, w_NN => prediction given by FNN
    I_app      => Inputs of the network
    t          => linspace where the solutions is evaluated
    indx       => which examples to plot
    """
    n_examples = len(indx)

    I_app = I_app[indx, :]

    I_app_max = np.max(I_app) * 1.2

    V_exact = V[indx, :]
    w_exact = w[indx, :]

    V_pred = V_NN[indx, :]
    w_pred = w_NN[indx, :]

    err_V = np.abs(V_exact - V_pred)
    err_w = np.abs(w_exact - w_pred)

    err_V_max = np.max(err_V) * 1.2
    err_V_min = np.min(err_V) * 1.2

    err_w_max = np.max(err_w) * 1.2
    err_w_min = np.min(err_w) * 1.2
    # V_pred = predicted[0,:,:]
    # w_pred = predicted[1,:,:]

    V_pred_max = np.max(V_pred) * 1.2
    V_pred_min = np.min(V_pred) * 1.2

    w_pred_max = np.max(w_pred) * 1.2
    w_pred_min = np.min(w_pred) * 1.2

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

        axs[2, i].semilogy(t, err_V[i, :], linewidth=0.9)
        axs[2, i].axes.set_xticklabels([])
        axs[2, i].set_ylim([err_V_min, err_V_max])
        axs[2, i].grid()

        axs[3, i].plot(t, w_exact[i, :], "b", linewidth=1.2)
        axs[3, i].plot(t, w_pred[i, :], "r--", linewidth=1.2)
        axs[3, i].axes.set_xticklabels([])
        axs[3, i].set_ylim([w_pred_min, w_pred_max])
        axs[3, i].grid()

        axs[4, i].semilogy(t, err_w[i, :], linewidth=0.9)
        axs[4, i].set_ylim([err_w_min, err_w_max])
        axs[4, i].grid()
        axs[4, i].set_xlabel("Time (ms)")
        if (
            i == 0
        ):  # just print the label, and for the first plot i set the label for the legend
            axs[0, i].set_ylabel("Current applied")
            axs[1, i].set_ylabel("V (mV)")
            axs[2, i].set_ylabel(r"$|V - V_{{\Theta}}|$")
            axs[3, i].set_ylabel("w")
            axs[4, i].set_ylabel(r"$|w - w_{{\Theta}}|$")
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
