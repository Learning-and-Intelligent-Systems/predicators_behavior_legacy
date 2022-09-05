"""Checks BEHAVIOR oracle ran tests with the given TIMEOUT by finding by
scraping the logs dir."""

import glob

import dill as pkl

tasks_to_test = [
    'collecting_aluminum_cans', 'throwing_away_leftovers',
    'packing_bags_or_suitcase', 'packing_boxes_for_household_move_or_trip',
    'opening_presents', 'organizing_file_cabinet', 'locking_every_window',
    'packing_car_for_trip', 're-shelving_library_books', 'storing_food',
    'organizing_boxes_in_garage', 'putting_leftovers_away',
    'unpacking_suitcase', 'putting_away_toys', 'boxing_books_up_for_storage',
    'sorting_books', 'clearing_the_table_after_dinner', 'opening_packages',
    'picking_up_take-out_food', 'collect_misplaced_items',
    'locking_every_door', 'putting_dishes_away_after_cleaning',
    'picking_up_trash', 'packing_food_for_work'
]

TIMEOUT = 1000

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
