import subprocess
import sys


def main():

    scripts = [
        "src/01_etl_sql.py",
        "src/02_database_merge.py",
        "src/02_eda_profiling.py",
        "src/03_analytics.py",
        "src/03_cohort_analysis.py",
        "src/04_visualization.py",
        "src/04_ml_modeling.py"
    ]

    for script in scripts:
        print(f"\nRunning {script}...")

        subprocess.run(
            [sys.executable, script],
            check=True
        )

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
