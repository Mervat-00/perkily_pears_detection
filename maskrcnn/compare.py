# compare.py
# Compare ALL runs from both Mask R-CNN and YOLOv11 in one table
# Run anytime: python compare.py

import json
import os

LOG_FILE = "evaluation_log.json"


def compare_runs():
    if not os.path.exists(LOG_FILE):
        print("No evaluation log found. Run evaluate.py or evaluate_yolo.py first.")
        return

    with open(LOG_FILE, 'r') as f:
        log = json.load(f)

    runs = log['runs']
    if len(runs) == 0:
        print("No runs saved yet.")
        return

    print(f"\n{'='*95}")
    print(f"  COMPARISON OF ALL RUNS  ({len(runs)} total)")
    print(f"{'='*95}")

    print(f"{'Run':<5} {'Date':<20} {'Model':<25} {'Epochs':<8} "
          f"{'ValLoss':<10} {'Avg F1':>8} {'Avg P':>8} {'Avg R':>8}  Notes")
    print("-" * 95)

    best_f1  = 0
    best_run = None

    for run in runs:
        avg       = run['metrics']['average']
        t_info    = run['training_info']
        f1        = avg['f1']
        model_type = t_info.get('model_type', 'unknown')[:24]
        marker    = " ⭐" if f1 > best_f1 else ""

        if f1 > best_f1:
            best_f1  = f1
            best_run = run

        print(f"#{run['run_id']:<4} "
              f"{run['timestamp']:<20} "
              f"{model_type:<25} "
              f"{str(t_info['epochs_trained']):<8} "
              f"{str(t_info['best_val_loss']):<10} "
              f"{avg['f1']:>8.2%} "
              f"{avg['precision']:>8.2%} "
              f"{avg['recall']:>8.2%}  "
              f"{run['notes']}{marker}")

    print("=" * 95)
    print(f"\n⭐ Best run: #{best_run['run_id']} "
          f"({best_run['notes']}) "
          f"— Avg F1: {best_f1:.2%}")

    # Per-class breakdown of best run
    print(f"\n{'='*60}")
    print(f"  BEST RUN BREAKDOWN  (Run #{best_run['run_id']})")
    print(f"  Model: {best_run['training_info'].get('model_type', 'unknown')}")
    print(f"{'='*60}")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for class_name, m in best_run['metrics'].items():
        print(f"{class_name:<15} {m['precision']:>10.2%} "
              f"{m['recall']:>10.2%} {m['f1']:>10.2%}")
    print("=" * 60)

    # F1 per class across all runs
    print(f"\n{'='*60}")
    print("  F1 SCORE PER CLASS ACROSS ALL RUNS")
    print(f"{'='*60}")
    class_names = ['Middle-Ripe', 'Ripe', 'Unripe', 'average']
    print(f"{'Class':<15}", end="")
    for run in runs:
        print(f"  Run #{run['run_id']:>2}", end="")
    print()
    print("-" * 60)
    for class_name in class_names:
        print(f"{class_name:<15}", end="")
        for run in runs:
            f1 = run['metrics'].get(class_name, {}).get('f1', 0)
            print(f"  {f1:>7.2%}", end="")
        print()
    print("=" * 60)


if __name__ == '__main__':
    compare_runs()
