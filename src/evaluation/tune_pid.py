import os
import json
import numpy as np
import time

from src.control.pid import PIDController


def simulate_step_response(Kp, Ki, Kd, target=1.0, steps=100):
    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd,
                        output_limits=(-30, 30))
    error = target
    errors = []
    outputs = []
    for _ in range(steps):
        output = pid.update(error)
        errors.append(error)
        outputs.append(output)
        error = target - output * 0.1
        if error < 0.01:
            error = 0
    return errors, outputs


def measure_settling_time(errors, threshold=0.05):
    for i, e in enumerate(errors):
        if all(abs(x) < threshold for x in errors[i:]):
            return i
    return len(errors)


def tune(output_dir: str = "reports"):
    os.makedirs(output_dir, exist_ok=True)

    param_grid = {
        "Kp": [0.5, 1.0, 2.0, 3.0, 4.0],
        "Ki": [0.0, 0.05, 0.1],
        "Kd": [0.0, 0.5, 1.0],
    }

    all_results = []

    for Kp in param_grid["Kp"]:
        for Ki in param_grid["Ki"]:
            for Kd in param_grid["Kd"]:
                errors, outputs = simulate_step_response(Kp, Ki, Kd)
                settling = measure_settling_time(errors)
                overshoot = max(0, max(errors) - 0)
                steady_state = np.mean(errors[-10:]) if len(errors) >= 10 else errors[-1]

                result = {
                    "Kp": Kp, "Ki": Ki, "Kd": Kd,
                    "settling_time_steps": settling,
                    "overshoot": float(overshoot),
                    "steady_state_error": float(abs(steady_state)),
                }
                all_results.append(result)

    best = min(all_results, key=lambda r: r["settling_time_steps"])

    report = {
        "sweep_results": all_results,
        "best_params": {
            "Kp": best["Kp"],
            "Ki": best["Ki"],
            "Kd": best["Kd"],
            "settling_time_steps": best["settling_time_steps"],
            "overshoot": best["overshoot"],
        },
    }

    report_path = os.path.join(output_dir, "logs", "pid_tuning.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"PID tuning saved to {report_path}")
    print(f"Best: Kp={best['Kp']}, Ki={best['Ki']}, Kd={best['Kd']}, "
          f"Settling={best['settling_time_steps']} steps")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for idx, (Kp_val, ax) in enumerate(zip([0.5, 1.0, 2.0, 4.0],
                                                axes.flat)):
            for Kd_val in [0.0, 0.5, 1.0]:
                errors, _ = simulate_step_response(Kp_val, 0.05, Kd_val)
                ax.plot(errors, label=f"Kd={Kd_val}")
            ax.set_title(f"Kp={Kp_val}, Ki=0.05")
            ax.set_xlabel("Step")
            ax.set_ylabel("Error")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle("PID Step Response", fontsize=14)
        fig_path = os.path.join(output_dir, "figures", "pid_step_response.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        print(f"Step response plot saved to {fig_path}")
        plt.close()
    except ImportError:
        print("Matplotlib not available, skipping plot.")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    tune(args.output)
