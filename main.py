import argparse
import sys
import os

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader  import load_or_generate, COMPOUND_LABELS, TEAM_ENCODING
from src.model  import split_data, train, save_model, predict_single
from src.evaluate     import (
    evaluate,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_decision_tree,
    plot_class_distribution,
)


BANNER = r"""
 ███████╗ ██╗     ███████╗████████╗ █████╗ ██████╗ ████████╗      █████╗ ██╗
 ██╔════╝███║     ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝     ██╔══██╗██║
 █████╗  ╚██║     ███████╗   ██║   ███████║██████╔╝   ██║   █████╗███████║██║
 ██╔══╝   ██║     ╚════██║   ██║   ██╔══██║██╔══██╗   ██║   ╚════╝██╔══██║██║
 ██║      ██║     ███████║   ██║   ██║  ██║██║  ██║   ██║         ██║  ██║██║
 ╚═╝      ╚═╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝         ╚═╝  ╚═╝╚═╝
           Predictive Modeling for F1 Initial Race Strategy
"""


def parse_args():
    parser = argparse.ArgumentParser(description="F1Start-AI Pipeline")
    parser.add_argument("--csv",      type=str,  default=None,
                        help="Path to real F1 CSV data (optional)")
    parser.add_argument("--no-tune",  action="store_true",
                        help="Skip GridSearchCV hyperparameter tuning")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    return parser.parse_args()


def demo_predictions(clf):
    """Run 5 hand-crafted scenarios to showcase the model."""
    print("\n" + "─" * 60)
    print("  DEMO: Single-Race Strategy Predictions")
    print("─" * 60)

    scenarios = [
        {"name": "Verstappen – P1, hot Bahrain",
         "grid_pos": 1, "track_temp": 52.0, "air_temp": 38.0, "rain_prob": 0, "team": 1},
        {"name": "Leclerc – P3, cool Monza",
         "grid_pos": 3, "track_temp": 28.0, "air_temp": 22.0, "rain_prob": 0, "team": 2},
        {"name": "Hamilton – P6, warm Silverstone",
         "grid_pos": 6, "track_temp": 37.5, "air_temp": 27.0, "rain_prob": 0, "team": 3},
        {"name": "Back-marker – P18, hot track",
         "grid_pos": 18, "track_temp": 48.0, "air_temp": 35.0, "rain_prob": 0, "team": 9},
        {"name": "Midfield – P12, mild conditions",
         "grid_pos": 12, "track_temp": 33.0, "air_temp": 25.0, "rain_prob": 0, "team": 6},
    ]

    for s in scenarios:
        result = predict_single(
            clf, s["grid_pos"], s["track_temp"],
            s["air_temp"], s["rain_prob"], s["team"],
        )
        print(f"\n  {s['name']}")
        print(f"    → Predicted Compound : {result['predicted_compound']}  "
              f"(confidence {result['confidence']})")
        print(f"    → Probabilities      : {result['probabilities']}")

    print("\n" + "─" * 60 + "\n")


def main():
    print(BANNER)
    args = parse_args()

    # 1. Load / generate data
    df = load_or_generate(args.csv)

    #2. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Train 
    clf = train(X_train, y_train, tune_hyperparams=not args.no_tune)

    # 4. Evaluate
    results = evaluate(clf, X_test, y_test)

    #5. Plots
    if not args.no_plots:
        plot_confusion_matrix(y_test, results["y_pred"])
        plot_feature_importance(clf)
        plot_decision_tree(clf, max_depth=3)
        plot_class_distribution(df)
        print("[Main] All plots saved to outputs/")

    #6. Save model 
    save_model(clf)

    #7. Demo predictions 
    demo_predictions(clf)

    print(f"[Main] Pipeline complete. Final test accuracy: "
          f"{results['accuracy'] * 100:.2f}%\n")

if __name__ == "__main__":
    main()
    

"""
Usage:
    python main.py  :-Full pipeline (train + evaluate + demo)
    python main.py --csv data/raw/f1_strategy.csv  :-Use real CSV data
    python main.py --no-tune  :-Skip GridSearchCV (faster)
"""
