from __future__ import annotations
import os
import matplotlib.pyplot as plt

SAMPLES = [160, 320, 640]

DATA = {
    "-O0": {
        "func_A": {"vuln": [0.70, 0.78, 0.82], "patch": [0.68, 0.76, 0.81]},
        "func_B": {"vuln": [0.55, 0.62, 0.66], "patch": [0.53, 0.60, 0.65]},
    },
    "-O1": {
        "func_A": {"vuln": [0.60, 0.66, 0.70], "patch": [0.58, 0.65, 0.69]},
        "func_B": {"vuln": [0.50, 0.56, 0.60], "patch": [0.49, 0.55, 0.59]},
    },
    "-O2/-O3": {
        "func_A": {"vuln": [0.62, 0.68, 0.72], "patch": [0.61, 0.67, 0.71]},
        "func_B": {"vuln": [0.52, 0.58, 0.61], "patch": [0.51, 0.57, 0.60]},
    },
}


def _validate():
    if not DATA:
        raise ValueError("DATA is empty.")
    for opt, funcs in DATA.items():
        if not funcs:
            raise ValueError(f"No functions under {opt}.")
        for fn, series in funcs.items():
            for variant, vals in series.items():
                if len(vals) != len(SAMPLES):
                    raise ValueError(
                        f"Length mismatch for {opt}/{fn}/{variant}: "
                        f"expected {len(SAMPLES)} got {len(vals)}"
                    )


def _collect_series_names() -> list[str]:
    names: list[str] = []
    for opt, funcs in DATA.items():
        for fn, series in funcs.items():
            for variant in series.keys():
                s = f"{fn}_{variant}"
                if s not in names:
                    names.append(s)
    return names


def _safe_opt_name(opt: str) -> str:
    """
    Make filenames that match your LaTeX:
      -O0      -> O0
      -O1      -> O1
      -O2/-O3  -> O23
    """
    if opt.strip() == "-O2/-O3":
        return "O23"
    return opt.replace("-", "").replace("/", "").replace(" ", "")


def plot_each_opt_in_its_own_window(
    save_dir: str = "figs",          # <-- default saves automatically
    show: bool = True,
    share_y_limits: bool = True,
):
    _validate()
    os.makedirs(save_dir, exist_ok=True)  # <-- ensure folder exists

    opt_levels = list(DATA.keys())
    series_names = _collect_series_names()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    series_to_color = {s: color_cycle[i % len(color_cycle)] for i, s in enumerate(series_names)}

    if share_y_limits:
        all_vals = []
        for opt in opt_levels:
            for fn, series in DATA[opt].items():
                for _, yvals in series.items():
                    all_vals.extend(yvals)
        ymin, ymax = min(all_vals), max(all_vals)
        pad = 0.03 * (ymax - ymin if ymax > ymin else 1.0)
        global_ylim = (ymin - pad, ymax + pad)
    else:
        global_ylim = None

    for opt in opt_levels:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        try:
            fig.canvas.manager.set_window_title(f"Table IV — {opt}")
        except Exception:
            pass

        ax.set_title(opt)
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Detection / differentiation accuracy")
        ax.set_xticks(SAMPLES)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

        if global_ylim is not None:
            ax.set_ylim(*global_ylim)

        handles_by_label = {}
        for fn, series in DATA[opt].items():
            for variant, yvals in series.items():
                label = f"{fn}_{variant}"
                c = series_to_color[label]
                line, = ax.plot(SAMPLES, yvals, color=c, marker="o", linewidth=1.8, label=label)
                handles_by_label[label] = line

        ordered_labels = [s for s in series_names if s in handles_by_label]
        ax.legend(
            [handles_by_label[l] for l in ordered_labels],
            ordered_labels,
            title="Series (color)",
            loc="best",
            frameon=True,
        )

        fig.tight_layout()

        # <-- always save
        safe_opt = _safe_opt_name(opt)
        out_path = os.path.join(save_dir, f"table_iv_{safe_opt}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

        if show:
            plt.show()

        plt.close(fig)


if __name__ == "__main__":
    plot_each_opt_in_its_own_window(show=True, share_y_limits=True)