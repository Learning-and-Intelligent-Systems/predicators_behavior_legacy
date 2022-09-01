import glob
import dill as pkl

for filename in glob.glob("logs/*/*"):
    with open(filename, "rb") as f:
        data = pkl.load(f)
    if data['results']['num_solved'] != 0:
        print(filename)
