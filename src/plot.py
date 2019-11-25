import numpy as np 
import matplotlib.pyplot as plt

def getstats(data, maxlen, logscale=False):
    stat_avg = []
    stat_min, stat_max = [], []
    for i in range(maxlen):
        cnt = 0
        pavg, pmin, pmax = 0.0, np.inf, -np.inf
        for exp in data:
            if i >= len(exp):
                continue
            cnt += 1
            pavg += exp[i]
            pmin = min(pmin, exp[i])
            pmax = max(pmax, exp[i])
        pavg /= cnt
        stat_avg.append(pavg)
        stat_max.append(pmax)
        stat_min.append(pmin)
    if not logscale:
        return stat_avg, stat_max, stat_min
    else:
        return np.log10(stat_avg), np.log10(stat_max), np.log10(stat_min)

def plot(root, method, wavg, line, lambd, color, facecolor, plottype="primal", expnum=10, logscale=False, label=""):

    primals = []
    duals = []
    test_errors = []

    maxlen = 0
    for expidx in range(expnum):
        # name = "{}/{}_{}.npz".format(root, "bcfw", expidx)
        name = "{}/{}_{}_{}_{:.4f}_{}.npz".format(root, method, "wavg" if wavg else "nowavg", 
            "line" if line else "noline", lambd, expidx)

        data = np.load(name)
        primals.append(data['primal'])
        if 'dual' in data:
            duals.append(data['dual'])
        test_errors.append(data['test_error'])

        if len(data['eff_pass']) > maxlen:
            maxlen = len(data['eff_pass'])
            eff_pass = data['eff_pass']

    x = eff_pass

    if plottype == "primal":
        data_avg, data_max, data_min = getstats(primals, maxlen, logscale)
    elif plottype == "testerror":
        data_avg, data_max, data_min = getstats(test_errors, maxlen, logscale)
    else:
        raise NotImplementedError("?")

    plt.plot(x, data_avg, '--', color=color, label=label)
    plt.fill_between(x, data_min, data_max, facecolor=facecolor, alpha=0.4, edgecolor='none')

def fix_lambda(root, lambd, expnum):

    plt.clf()
    plot(root, "bcfw", True, True, lambd, 'blue', 'blue', plottype="primal", expnum=expnum, logscale=True, label="bcfw (line search)")
    plot(root, "bcfw", False, True, lambd, 'cyan', 'cyan', plottype="primal", expnum=expnum, logscale=True, label="bcfw (no wavg, line search)")
    plot(root, "bcfw", False, False, lambd, 'royalblue', 'royalblue', plottype="primal", expnum=expnum, logscale=True, label="bcfw (no wavg, gamma)")

    plot(root, "fw", False, True, lambd, 'orange', 'orange', plottype="primal", expnum=expnum, logscale=True, label="fw (line search)")
    plot(root, "fw", False, False, lambd, 'red', 'red', plottype="primal", expnum=expnum, logscale=True, label="fw (gamma)")

    plot(root, "ssg", True, False, lambd, 'g', 'g', plottype="primal", expnum=expnum, logscale=True, label="ssg")
    plot(root, "ssg", False, False, lambd, 'lime', 'lime', plottype="primal", expnum=expnum, logscale=True, label="ssg (no wavg)")

    plt.xlabel("eff_pass")
    plt.ylabel("primal suboptimality in logscale")

    plt.ylim(-1, 5)

    plt.savefig("lambda_{:.4f}.png".format(lambd))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot("../output", "bcfw", True, True, 1e-2, 'r', 'r', plottype="primal", expnum=10, logscale=True, label="bcfw")
    fix_lambda("../new_output", 1e-2, 5)

