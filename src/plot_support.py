from typing import List

import matplotlib.pyplot as plt
import numpy as np
import wandb


LAMBDA = 1.3 * 11 * 10000
X_MAX = 20

RUN_PATHS = {
    "net-fragments": "sagerpascal/net-fragments-final/m8ecvdhb",
}



def limit_support(x: List[float]) -> List[float]:
    """
    Limit the support of the lateral connections.
    :param x: Support per cell
    :return: Limited lateral connections
    """
    return [min(x_i, LAMBDA) for x_i in x]


def get_support_from_wandb(run_id: str) -> (List[float], List[float], List[float], List[float]):
    """
    Get support from wandb.
    :param run_id: Run ID
    :return: Tuple of lists of support values
    """
    # Get the run's history
    api = wandb.Api()
    run = api.run(run_id)
    history = run.scan_history()

    # Get the support values
    avg_support_active = [h['S2/avg_support_active'] for h in history if h['S2/avg_support_active'] is not None]
    min_support_active = [h['S2/min_support_active'] for h in history if h['S2/min_support_active'] is not None]
    max_support_active = [h['S2/max_support_active'] for h in history if h['S2/max_support_active'] is not None]
    avg_support_inactive = [h['S2/avg_support_inactive'] for h in history if h['S2/avg_support_inactive'] is not None]

    # in the code, the support is limited to lambda -> consider this for this plot!
    avg_support_active = limit_support(avg_support_active)
    min_support_active = limit_support(min_support_active)
    max_support_active = limit_support(max_support_active)
    avg_support_inactive = limit_support(avg_support_inactive)

    return avg_support_active, min_support_active, max_support_active, avg_support_inactive

def print_support_active_cells(title, min_support_active, max_support_active, avg_support_active, avg_support_inactive):
    """
    Plot the support of fragments cells.
    :param title: Title of the plot
    :param min_support_active: Min support active cells
    :param max_support_active: Max support active cells
    :param avg_support_active: Avg support active cells
    :param avg_support_inactive: Avg support inactive cells
    """

    # if problems with font, run it on local machine
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Helvetica"
    x = np.arange(1, len(avg_support_active) + 1, 1)

    fig, ax = plt.subplots(dpi=300, figsize=(8, 4))
    # ax.plot(x, avg_support_active, label="avg. support active cells (without inhibition)")
    # ax.fill_between(x, min_support_active, max_support_active, color='b', alpha=.15,
    #                 label="min/max support (without inhibition)")

    ax.plot(x, avg_support_active, label="avg. support active cells", color='b')
    ax.fill_between(x, min_support_active, max_support_active, color='b', alpha=.15,
                    label="min/max support")

    # ax.plot(x, [LAMBDA] * len(avg_support_active), color='r', linestyle='--', label="λ")
    ax.plot(x, avg_support_inactive, label="avg. support inactive cells", color='orange')
    # plt.title(title)
    plt.legend()
    plt.ylabel("Support Strength")
    plt.xlabel("Epoch")
    plt.yticks(np.arange(0, X_MAX + 1, 2), np.arange(0, X_MAX + 1, 2))
    d = 2 if len(avg_support_active) <= 50 else 4
    plt.xticks(np.arange(0, len(avg_support_active) + 1, d), np.arange(0, len(avg_support_active) + 1, d))
    plt.xlim(1, len(avg_support_active))
    plt.ylim(0, X_MAX)
    plt.tight_layout()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    for title, run_id in RUN_PATHS.items():
        avg_support_active, min_support_active, max_support_active, avg_support_inactive = get_support_from_wandb(run_id)
        print_support_active_cells(title, min_support_active, max_support_active, avg_support_active, avg_support_inactive)
