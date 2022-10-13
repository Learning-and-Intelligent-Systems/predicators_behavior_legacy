import json

path_to_file = \
                "predicators/behavior_utils/task_to_preselected_scenes.json"
with open(path_to_file, 'rb') as f:
    task_to_preselected_scenes = json.load(f)
tasks = ['collecting_aluminum_cans', 'throwing_away_leftovers', 'setting_up_candles', 'packing_bags_or_suitcase', 'packing_boxes_for_household_move_or_trip', 'opening_presents', 'assembling_gift_baskets', 'organizing_file_cabinet', 'locking_every_window', 'packing_car_for_trip', 're-shelving_library_books', 'storing_food', 'organizing_boxes_in_garage', 'polishing_silver', 'putting_leftovers_away', 'unpacking_suitcase', 'putting_away_toys', 'boxing_books_up_for_storage', 'packing_adult_s_bags', 'sorting_books', 'clearing_the_table_after_dinner', 'opening_packages', 'laying_tile_floors', 'bringing_in_wood', 'picking_up_take-out_food', 'loading_the_dishwasher', 'cleaning_freezer', 'cleaning_the_pool', 'collect_misplaced_items', 'locking_every_door', 'putting_dishes_away_after_cleaning', 'filling_a_Christmas_stocking', 'picking_up_trash', 'packing_lunches', 'cleaning_a_car', 'cleaning_garage', 'moving_boxes_to_storage', 'packing_food_for_work']

for task in tasks:
    scene = task_to_preselected_scenes[task][0]
    yaml_str = f"""  {task}-{scene}:
    NAME: "behavior"
    FLAGS:
      behavior_scene_name: {scene}
      behavior_task_list: "[{task}]"
      behavior_option_model_eval: True"""
    print(yaml_str)
