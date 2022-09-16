"""Checks BEHAVIOR oracle ran tests with the given TIMEOUT by finding by
scraping the logs dir."""

import glob

import dill as pkl

tasks_to_test = [
    're-shelving_library_books',
    'collecting_aluminum_cans',
    'throwing_away_leftovers',
    'opening_presents',
    'locking_every_door',
    'locking_every_window',
    'opening_packages',
    'sorting_books'
]

TIMEOUT = 500

# Globs logs for tasks with results file.
tasks = []
for filename in glob.glob(f"logs/*{TIMEOUT}/*"):
    with open(filename, "rb") as f:
        data = pkl.load(f)
    # Get task name.
    task = ""
    for c in filename.split('logs/')[-1]:
        if c.isupper():
            break
        task += c
    task = task[:-1]
    # Append whether task was solved or not.
    if data['results']['num_solved'] != 0:
        tasks.append([task, True])
    else:
        tasks.append([task, False])

solved_tasks = [task[0] for task in tasks if task[1]]
failed_tasks = [task[0] for task in tasks if not task[1]]

print("Solved", len(solved_tasks), "/", len(tasks))
print("Solved tasks", solved_tasks)
print()
print("Unique Solved", len(set(solved_tasks)), "/",
      len(set([task[0] for task in tasks])))
print("Unique Solved tasks", set(solved_tasks))
print()
print("Failed", len(failed_tasks), "/", len(tasks))
print("Failed tasks", failed_tasks)
print()
print("Task with Errors",
      len(tasks_to_test) * 3 - len(solved_tasks + failed_tasks), "/",
      len(tasks_to_test) * 3)
print("Unique Task with Errors",
      len(tasks_to_test) - len(set(solved_tasks + failed_tasks)), "/",
      len(tasks_to_test))
