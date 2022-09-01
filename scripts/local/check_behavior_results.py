"""Given BEHAVIOR experiments have been run with the `run_behavior_tests`
script, analyze the logs directory and print all tasks that have a non-zero
success rate."""

import glob

import dill as pkl


def _main() -> None:
    for filename in glob.glob("logs/*/*"):
        with open(filename, "rb") as f:
            data = pkl.load(f)
        if data['results']['num_solved'] != 0:
            print(filename)


if __name__ == '__main__':
    _main()
