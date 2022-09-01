import os
import json


NUM_TEST = 1
SEED = 0
TIMEOUT = 100

path_to_file = "predicators/behavior_utils/task_to_preselected_scenes.json"
f = open(path_to_file)
data = json.load(f)
f.close()

tasks_to_test = ['collecting_aluminum_cans', 'throwing_away_leftovers', 'packing_bags_or_suitcase', 'packing_boxes_for_household_move_or_trip', 'opening_presents', 'organizing_file_cabinet', 'locking_every_window', 'packing_car_for_trip', 're-shelving_library_books', 'storing_food', 'organizing_boxes_in_garage', 'putting_leftovers_away', 'unpacking_suitcase', 'putting_away_toys', 'boxing_books_up_for_storage', 'sorting_books', 'clearing_the_table_after_dinner', 'opening_packages', 'picking_up_take-out_food', 'collect_misplaced_items', 'locking_every_door', 'putting_dishes_away_after_cleaning', 'picking_up_trash', 'cleaning_a_car', 'packing_food_for_work']

# Create commands to run
cmds = []
for task, scenes in data.items():
    if task in tasks_to_test:
        for scene in scenes:
            logfolder = os.path.join("logs", f"{task}_{scene}_{SEED}"
                                        f"_{NUM_TEST}_{TIMEOUT}/")
            try:
                os.mkdir(logfolder)
            except OSError:
                os.rmdir(logfolder)
                os.mkdir(logfolder)
            
            cmds.append("python predicators/main.py "
                        "--env behavior "
                        "--approach oracle "
                        "--behavior_mode headless "
                        "--option_model_name oracle_behavior "
                        "--num_train_tasks 1 "
                        f"--num_test_tasks {NUM_TEST} "
                        f"--behavior_scene_name {scene} "
                        f"--behavior_task_name {task} "
                        f"--seed {SEED} "
                        f"--offline_data_planning_timeout {TIMEOUT} "
                        f"--timeout {NUM_TEST} "
                        "--behavior_option_model_eval True "
                        "--plan_only_eval True "
                        f"--results_dir {logfolder}")

# Run the commands in order.
num_cmds = len(cmds)
for i, cmd in enumerate(cmds):
    print(f"********* RUNNING COMMAND {i+1} of {num_cmds} *********")
    result = os.popen(cmd).read()

