"""Utility functions for using iGibson and BEHAVIOR."""

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pybullet as p

from predicators.settings import CFG
from predicators.structs import Array, GroundAtomTrajectory, \
    LowLevelTrajectory, Predicate, Set, State
from predicators.utils import abstract

try:
    from igibson.envs.behavior_env import \
        BehaviorEnv  # pylint: disable=unused-import
    from igibson.object_states.on_floor import \
        RoomFloor  # pylint: disable=unused-import
    from igibson.objects.articulated_object import URDFObject
    from igibson.robots.behavior_robot import \
        BRBody  # pylint: disable=unused-import
    from igibson.robots.robot_base import \
        BaseRobot  # pylint: disable=unused-import
    from igibson.utils.checkpoint_utils import load_checkpoint
except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
    pass

# NOTE: Each of these below constants obtained from parsing script in
# LIS fork of the bddl repository. These sets might be incomplete, so
# if you encounter an error while trying to solve a new BEHAVIOR task,
# you might need to add to these.
ALL_RELEVANT_OBJECT_TYPES = {
    'breakfast_table', 'notebook', 'hardback', 'shelf', 'agent', 'room_floor',
    'coffee_table', 'pop', 'bed', 'bucket', 'plate', 'hamburger', 'countertop',
    'trash_can', 'backpack', 'toothbrush', 'shampoo', 'underwear', 'door',
    'window', 'toothpaste', 'package', 'highlighter', 'swivel_chair',
    'document', 'bottom_cabinet_no_top', 'folder', 'bottom_cabinet',
    'top_cabinet', 'sofa', 'bath_towel', 'bowl', 'bowl', 'cup', 'cup', 'sink'
}
PICK_PLACE_OBJECT_TYPES = {
    'mineral_water', 'oatmeal', 'blueberry', 'headset', 'jug', 'flank',
    'baseball', 'crab', 'dressing', 'cranberry', 'trout', 'kale', 'shoe',
    'licorice', 'decaffeinated_coffee', 'cookie', 'whiskey', 'bench',
    'alcohol', 'journal', 'fork', 'cherry', 'tin', 'winter_melon', 'cocktail',
    'pretzel', 'bidet', 'bowl', 'nectarine', 'dish', 'baked_goods', 'noodle',
    'gingerbread', 'lemonade', 'basket', 'sock', 'brandy', 'apricot', 'plum',
    'diary', 'cabbage', 'cupcake', 'lentil', 'turnip', 'solanaceous_vegetable',
    'straight_chair', 'pomegranate', 'pastry', 'sugar', 'marble', 'poultry',
    'backpack', 'tank', 'helmet', 'cornbread', 'ribbon', 'pitcher', 'caramel',
    'folding_chair', 'hamper', 'sushi', 'hot_sauce', 'cheeseboard', 'scoop',
    'highchair', 'mascara', 'newspaper', 'punch', 'liqueur', 'beef', 'jersey',
    'julienne', 'marker', 'root_vegetable', 'cantaloup', 'screwdriver',
    'softball', 'porterhouse', 'lipstick', 'highlighter', 'soap', 'candle',
    'ring', 'ale', 'banana', 'toothpaste', 'coffeepot', 'toothbrush',
    'carving_knife', 'meat_loaf', 'plaything', 'cake', 'bagel',
    'freshwater_fish', 'lobster', 'asparagus', 'quick_bread', 'olive', 'date',
    'sweatshirt', 'sprout', 'candy', 'corn_chip', 'coconut', 'blush_wine',
    'carrot', 'grapefruit', 'mozzarella', 'prawn', 'parmesan', 'dentifrice',
    'papaya', 'frying_pan', 'hardback', 'head_cabbage', 'potholder', 'currant',
    'mostaccioli', 'coffee', 'bracelet', 'flatbread', 'broccoli', 'dipper',
    'mocha', 'tea', 'loaf_of_bread', 'rump', 'plate', 'vidalia_onion',
    'eyeshadow', 'mandarin', 'painting', 'footstool', 'chip', 'apple',
    'tenderloin', 'sack', 'orange_liqueur', 'cucumber', 'snowball', 'lime',
    'rocking_chair', 'french_bread', 'stool', 'teapot', 'loafer', 'gumbo',
    'salami', 'granola', 'perfume', 'frisbee', 'alarm', 'chickpea', 'bun',
    'flour', 'stiletto', 'oxford', 'muskmelon', 'macaroni', 'armor_plate',
    'wine_bottle', 'mouse', 'lettuce', 'gouda', 'sausage', 'fennel', 'salmon',
    'crock', 'eggplant', 'potpourri', 'cardigan', 'saltwater_fish', 'pad',
    'cola', 'bead', 'venison', 'fruit_drink', 'blanc', 'coca_cola', 'brie',
    'butter', 'jewelry', 'pencil_box', 'anchovy', 'grape', 'cold_cereal',
    'dagger', 'cologne', 'chair', 'gazpacho', 'wine', 'shellfish', 'biscuit',
    'red_salmon', 'planner', 'bath', 'espresso', 'pepperoni', 'brush',
    'muffin', 'clamshell', 'baguet', 'penne', 'radish', 'cracker', 'chop',
    'pen', 'drum', 'mat', 'bathtub', 'edible_fruit', 'curacao', 'scotch',
    'sparkling_wine', 'pie', 'pineapple', 'tomato', 'sheet', 'gumdrop',
    'sharpie', 'pullover', 'smoothie', 'snack_food', 'cut_of_beef',
    'candy_cane', 'tablespoon', 'sofa', 'raspberry', 'pencil', 'coleslaw',
    'pancake', 'farfalle', 'tortilla', 'chili', 'spinach', 'umbrella', 'peach',
    'groundsheet', 'vegetable_oil', 'silver_salmon', 'mousetrap', 'feijoa',
    'clam', 'knife', 'tray', 'green_onion', 'armchair', 'crouton',
    'duffel_bag', 'cos', 'necklace', 'mattress', 'scanner', 'steak',
    'eyeliner', 'greens', 'zucchini', 'kiwi', 'game', 'tablefork', 'squash',
    'hamburger', 'marinara', 'lollipop', 'ball', 'juice', 'worcester_sauce',
    'chaise_longue', 'demitasse', 'cider', 'document', 'cauliflower', 'fish',
    'beverage', 'vessel', 'earplug', 'percolator', 'french_dressing',
    'sweet_pepper', 'workwear', 'pan', 'mussel', 'shank', 'modem', 'turkey',
    'pepper', 'vodka', 'rib', 'scraper', 'summer_squash', 'honeydew',
    'facsimile', 'pear', 'ravioli', 'dredging_bucket', 'cheese',
    'spaghetti_sauce', 'gorgonzola', 'lingerie', 'hat', 'nightgown', 'bird',
    'artichoke', 'brownie', 'pea', 'ice_cube', 'hanger', 'walker', 'doll',
    'paper_towel', 'milk', 'lemon', 'mayonnaise', 'brew', 'parsley',
    'cellophane', 'jar', 'broth', 'pop', 'champagne', 'wafer', 'rum',
    'soft_drink', 'barrel', 'water', 'towel', 'caldron', 'carafe', 'stockpot',
    'cream_pitcher', 'salad', 'kettle', 'olive_oil', 'cut_of_pork', 'book',
    'bow', 'chocolate', 'basin', 'apparel', 'autoclave', 'watermelon',
    'mushroom', 'soup', 'bacon', 'basketball', 'cube', 'telephone_receiver',
    'drinking_vessel', 'cup', 'fig', 'scrub_brush', 'bean', 'sandal',
    'dustpan', 'calculator', 'white_sauce', 'buttermilk', 'bath_towel',
    'plastic_wrap', 'chicory', 'dishrag', 'beet', 'boiler', 'roaster',
    'saucepan', 'bleu', 'cruet', 'radicchio', 'produce', 'nacho', 'tuna',
    'sirloin', 'veal', 'tea_bag', 'mug', 'ladle', 'lamb', 'mixed_drink',
    'sweater', 'lego', 'canola_oil', 'meat', 'orange', 'tortilla_chip',
    'fudge', 'shampoo', 'cheddar', 'breakfast_food', 'wrapping', 'mango',
    'nan', 'swivel_chair', 'catsup', 'bread', 'hand_towel', 'berry', 'makeup',
    'drinking_water', 'shirt', 'bell_pepper', 'pork', 'liquor', 'sandwich',
    'pumpkin', 'gym_shoe', 'vegetable', 'martini', 'melon', 'cayenne',
    'football', 'mint', 'album', 'beefsteak', 'seltzer', 'cognac', 'scrapbook',
    'white_bread', 'wreath', 'egg', 'notebook', 'cleansing_agent', 'rag',
    'beer', 'citrus', 'folder', 'bucket', 'detergent', 'chicory_escarole',
    'marshmallow', 'earphone', 'puppet', 'ham', 'dustcloth', 'briefcase',
    'prosciutto', 'chicken', 'gourd', 'tabasco', 'mural', 'printer', 'tequila',
    'potato', 'cut', 'gravy', 'scone', 'cereal', 'avocado', 'pasta', 'vase',
    'underwear', 'cheesecake', 'seafood', 'dried_fruit', 'shallot', 'carton',
    'legume', 'blackberry', 'tabbouleh', 'coffee_cup', 'salad_green',
    'paintbrush', 'brisket', 'sunhat', 'sunglass', 'toast', 'soy', 'seat',
    'carpet_pad', 'blender', 'tart', 'puff', 'jewel', 'strawberry', 'onion',
    'toasting_fork', 'bottle', 'pomelo', 'teacup', 'pot', 'food', 'bisque',
    'hairbrush', 'spoon', 'novel', 'teaspoon', 'waffle', 'pita', 'yogurt',
    'stout', 'dishtowel', 'paperback_book', 'casserole', 'earring',
    'peppermint', 'cruciferous_vegetable', 'soup_ladle', 'jean', 'teddy',
    'chestnut', 'sauce', 'piece_of_cloth', 'whitefish', 'siren', 'balloon',
    'celery', 'hot_pepper', 'raisin', 'sugar_jar', 'toy', 'sticky_note',
    't-shirt'
}
PLACE_ONTOP_SURFACE_OBJECT_TYPES = {
    'towel', 'tabletop', 'face', 'brim', 'cheddar', 'chaise_longue', 'stove',
    'gaming_table', 'rocking_chair', 'swivel_chair', 'car', 'dartboard',
    'hand_towel', 'edge', 'sheet', 'countertop', 'carton', 'deep-freeze',
    'shelf', 'floorboard', 'bookshelf', 'flatbed', 'worktable',
    'pedestal_table', 'highchair', 'side', 'sofa', 'armor_plate',
    'horizontal_surface', 'floor', 'mantel', 'table', 'gouda', 'clipboard',
    'breakfast_table', 'christmas_tree', 'dressing_table', 'basket',
    'mozzarella', 'cheese', 'truck_bed', 'screen', 'parmesan', 'gorgonzola',
    'pegboard', 'bath_towel', 'platform', 'writing_board', 'straight_chair',
    'coffee_table', 'desk', 'armchair', 'brie', 'front', 'bleu', 'helmet',
    'paper_towel', 'dishtowel', 'dial', 'folding_chair', 'deck', 'chair',
    'hamper', 'bed', 'plate', 'work_surface', 'board', 'pallet',
    'console_table', 'pool_table', 'electric_refrigerator', 'stand',
    'room_floor'
}
PLACE_INTO_SURFACE_OBJECT_TYPES = {
    'shelf', 'sack', 'basket', 'dredging_bucket', 'cabinet', 'crock', 'bucket',
    'casserole', 'bookshelf', 'teapot', 'dishwasher', 'deep-freeze', 'hamper',
    'bathtub', 'car', 'vase', 'jar', 'bin', 'mantel', 'stocking', 'ashcan',
    'electric_refrigerator', 'clamshell', 'backpack', 'sink', 'carton', 'dish',
    'trash_can', 'bottom_cabinet_no_top', 'fridge', 'bottom_cabinet',
    'top_cabinet'
}
OPENABLE_OBJECT_TYPES = {
    'sack', 'storage_space', 'trap', 'turnbuckle', 'lock', 'trailer_truck',
    'duplicator', 'slide_fastener', 'car', 'jug', 'percolator', 'window',
    'coupling', 'package', 'collar', 'deep-freeze', 'walnut', 'pack', 'truck',
    'nozzle', 'shoebox', 'journal', 'toolbox', 'grill', 'work', 'box', 'tin',
    'wine_bottle', 'canned_food', 'choke', 'file', 'door', 'bag', 'facsimile',
    'washer', 'envelope', 'basket', 'dredging_bucket', 'crock', 'spout',
    'carabiner', 'album', 'screen', 'diary', 'drain', 'watchband', 'armoire',
    'accessory', 'tent', 'dishwasher', 'dryer', 'vent', 'writing_board',
    'personal_computer', 'scrapbook', 'bale', 'bin', 'coil', 'egg', 'reactor',
    'belt', 'recreational_vehicle', 'canister', 'tie', 'notebook', 'backpack',
    'stopcock', 'brassiere', 'pencil_box', 'tank', 'marinade', 'material',
    'hinge', 'folder', 'wastepaper_basket', 'bucket', 'folding_chair',
    'digital_computer', 'jaw', 'drawstring_bag', 'hamper', 'caddy',
    'refrigerator', 'briefcase', 'shaker', 'hindrance', 'stapler', 'drawer',
    'jar', 'binder', 'planner', 'gear', 'cupboard', 'gourd', 'windowpane',
    'champagne', 'cooler', 'barrel', 'can', 'clamshell',
    'electric_refrigerator', 'van', 'bird_feeder', 'kit', 'plate_glass',
    'printer', 'mascara', 'velcro', 'pill', 'range_hood', 'packet', 'pen',
    'knot', 'stove', 'marker', 'wallet', 'wardrobe', 'faucet', 'polish',
    'cotter', 'bulldog_clip', 'connection', 'bundle', 'white_goods',
    'office_furniture', 'vase', 'carryall', 'lipstick', 'sparkling_wine',
    'kettle', 'highlighter', 'clamp', 'book', 'microwave', 'nutcracker',
    'banana', 'pane', 'hole', 'coffeepot', 'bow', 'carton', 'sharpie',
    'toilet', 'canopy', 'autoclave', 'pouch', 'drawstring', 'ventilation',
    'lid', 'caster', 'buckle', 'mail', 'portable_computer', 'clog', 'capsule',
    'clipboard', 'umbrella', 'packaging', 'laptop', 'drumstick', 'thing',
    'mousetrap', 'shoulder_bag', 'latch', 'movable_barrier', 'hardback',
    'chest_of_drawers', 'cage', 'novel', 'roaster', 'duffel_bag', 'diaper',
    'ashcan', 'junction', 'mechanical_system', 'crusher', 'frame', 'shelter',
    'chest', 'magazine', 'wicker', 'paperback_book', 'scanner', 'computer',
    'dose', 'clasp', 'eyeliner', 'clothespin', 'hood', 'trademark', 'pincer',
    'crate', 'cabinet', 'joint', 'bottom_cabinet_no_top', 'fridge',
    'bottom_cabinet', 'trash_can'
}


CLEANING_OBJECT_TYPES = {
    'toothbrush', 'towel', 'dinner_napkin', 'paper_towel', 'dishtowel', 'broom',
    'vacuum', 'rag', 'carpet_sweeper', 'hand_towel', 'scraper', 'bath_towel', 'eraser',
    'dustcloth', 'scrub_brush'
}

DUSTYABLE_OBJECT_TYPES = {'tabletop', 'face', 'dumbbell', 'corkscrew', 'terry', 'circle', 'fur', 'coaster', 'gauze', 'cotton', 'chock', 'trap', 'compact_disk', 'cap', 'stake', 'converter', 'peripheral', 'slide_fastener', 'headset', 'jug', 'baseball', 'window', 'straightener', 'computer_game', 'toaster', 'data_input_device', 'collar', 'accelerator', 'shoe', 'truck', 'remote_control', 'apron', 'nozzle', 'dander', 'bench', 'washcloth', 'ipod', 'journal', 'fork', 'outerwear', 'cherry', 'ammunition', 'work', 'quilt', 'wheel', 'tin', 'plywood', 'instrument', 'bidet', 'bag', 'bowl', 'squeegee', 'water_scooter', 'washer', 'panel', 'basket', 'motor_vehicle', 'mixer', 'hammer', 'suit', 'carabiner', 'sock', 'slipper', 'screen', 'boat', 'bath_linen', 'diary', 'drain', 'armoire', 'pegboard', 'dishwasher', 'vent', 'golf_equipment', 'wrench', 'personal_computer', 'straight_chair', 'suede_cloth', 'semiconductor_device', 'ginger', 'machine', 'backpack', 'tank', 'computer_circuit', 'helmet', 'magnetic_disk', 'ribbon', 'arrangement', 'pitcher', 'table_knife', 'disk', 'folding_chair', 'circuit', 'pump', 'toweling', 'solid_figure', 'drawstring_bag', 'rule', 'hamper', 'caddy', 'cheeseboard', 'console_table', 'headlight', 'medical_instrument', 'can', 'highchair', 'musical_instrument', 'canvas', 'velcro', 'mascara', 'loudspeaker', 'hacksaw', 'newspaper', 'packet', 'knot', 'jersey', 'shaver', 'screw', 'marker', 'floor_lamp', 'carriage', 'act', 'tulle', 'screwdriver', 'greatcoat', 'softball', 'cotter', 'parlor_game', 'skewer', 'tack', 'eiderdown', 'highlighter', 'candle', 'ring', 'bell', 'banana', 'grate', 'coffeepot', 'crib', 'flashlight', 'shrapnel', 'toothbrush', 'inverter', 'shelf', 'regulator', 'carving_knife', 'psychological_feature', 'camcorder', 'spring', 'razor', 'ceramic_ware', 'indentation', 'post', 'plumbing', 'handle', 'caster', 'staple', 'buckle', 'frill', 'tumbler', 'steamer', 'cradle', 'earmuff', 'digital_camera', 'swab', 'shield', 'hose', 'caliper', 'activity', 'mill', 'acoustic_device', 'room', 'carrot', 'drumstick', 'thimble', 'ladder', 'thing', 'snake', 'wedge', 'stairway', 'voltmeter', 'heater', 'push_button', 'frying_pan', 'photograph', 'hardback', 'basil', 'trouser', 'rail', 'folderal', 'junction', 'dress', 'anchor', 'blade', 'change_of_location', 'chest', 'bolt', 'bracelet', 'photographic_equipment', 'printed_circuit', 'dial', 'stringed_instrument', 'computer', 'clasp', 'dipper', 'cornice', 'uniform', 'vacuum', 'electro-acoustic_transducer', 'clothespin', 'hood', 'plate', 'bannister', 'brocade', 'pincer', 'pool_table', 'eyeshadow', 'painting', 'footstool', 'bedpost', 'power_tool', 'sifter', 'drum_sander', 'drive', 'blazer', 'layer', 'apple', 'shaft', 'writing', 'towel_rack', 'thermostat', 'microphone', 'rocking_chair', 'stool', 'teapot', 'turnbuckle', 'toothpick', 'stirrer', 'loafer', 'whisk', 'sieve', 'frisbee', 'movement', 'alarm', 'fan', 'dartboard', 'spatula', 'filter', 'gauge', 'bobbin', 'countertop', 'electrical_converter', 'heating_element', 'sink', 'stiletto', 'porcelain', 'product', 'roller', 'pack', 'circuit_breaker', 'oxford', 'projection', 'step_ladder', 'shoebox', 'protective_garment', 'antenna', 'worktable', 'sphere', 'armor_plate', 'box', 'adapter', 'disk_drive', 'digital_display', 'repeater', 'mantel', 'wine_bottle', 'baggage', 'pocketknife', 'diskette', 'basketball_equipment', 'shear', 'table', 'table_lamp', 'file', 'mouse', 'door', 'receiver', 'solid', 'cylinder', 'breakfast_table', 'radio_receiver', 'cinder', 'brick', 'crock', 'spout', 'blanket', 'background', 'watchband', 'tile', 'battery', 'handlebar', 'floor_cover', 'cord', 'bead', 'television_equipment', 'tablet', 'writing_board', 'faceplate', 'appendage', 'broomstick', 'coffee_table', 'desk', 'bale', 'model', 'hoop', 'belt', 'trimmer', 'recreational_vehicle', 'laundry', 'putter', 'drafting_instrument', 'stopcock', 'module', 'pencil_box', 'jewelry', 'yardstick', 'dolly', 'wastepaper_basket', 'dagger', 'liquid_crystal_display', 'digital_computer', 'well', 'chair', 'foot_rule', 'clothesline', 'set', 'purifier', 'lens', 'cpu_board', 'binder', 'planner', 'bath', 'upholstery', 'resistor', 'cooler', 'plate_glass', 'drill', 'van', 'clamshell', 'brush', 'diode', 'pill', 'drum', 'timer', 'case', 'mat', 'windowsill', 'wall_clock', 'gaming_table', 'wallet', 'composition', 'faucet', 'bathtub', 'air_pump', 'bundle', 'electronic_device', 'clout_nail', 'elastic_device', 'applicator', 'pineapple', 'opener', 'tomato', 'microwave', 'piano', 'nutcracker', 'upright', 'sleeve', 'lighting_fixture', 'pane', 'noisemaker', 'winder', 'pedal', 'broom', 'measuring_stick', 'hubcap', 'bulletin_board', 'canopy', 'pedestal_table', 'guard', 'tablespoon', 'sofa', 'ventilation', 'sail', 'floor', 'pencil', 'cringle', 'utility', 'breathing_device', 'silk', 'umbrella', 'laptop', 'boot', 'peach', 'electric_lamp', 'crowbar', 'groundsheet', 'funnel', 'shell', 'fitting', 'tire', 'group', 'nail', 'mousetrap', 'knife', 'tray', 'cellular_telephone', 'firebox', 'shoulder_bag', 'inflater', 'movable_barrier', 'baseboard', 'chest_of_drawers', 'piston', 'armchair', 'duffel_bag', 'speedometer', 'sculpture', 'baby_bed', 'sled', 'paper_fastener', 'necklace', 'happening', 'mattress', 'ceramic', 'wicker', 'scanner', 'sprocket', 'eyeliner', 'craft', 'shade', 'thermometer', 'memory_device', 'rail_fence', 'footboard', 'runner', 'paperweight', 'game', 'crate', 'tablefork', 'recording', 'cabinet', 'television_receiver', 'step', 'storage_space', 'crayon', 'crossbar', 'chaise_longue', 'griddle', 'demitasse', 'document', "plumber's_snake", 'converging_lens', 'lock', 'trailer_truck', 'compressor', 'plastic_art', 'car', 'display_panel', 'earplug', 'golf_club', 'blower', 'sharpener', 'percolator', 'percussion_instrument', 'charger', 'grater', 'vessel', 'cooking_utensil', 'manifold', 'exercise_device', 'blinker', 'reamer', 'rack', 'deep-freeze', 'timepiece', 'machinery', 'workwear', 'cable', 'fire_extinguisher', 'pan', 'floorboard', 'wallboard', 'bookshelf', 'vest', 'pestle', 'side', 'modem', 'gingham', 'display', 'motorcycle', 'written_communication', 'tarpaulin', 'flatware', 'doorknob', 'facsimile', 'dredging_bucket', 'cassette', 'opening', 'bicycle', 'truck_bed', 'plunger', 'light_bulb', 'tent', 'hat', 'dryer', 'protractor', 'library', 'wiring', 'tongs', 'burner', 'blackboard', 'bin', 'portrait', 'strengthener', 'steel', 'hanger', 'lath', 'string', 'walker', 'doll', 'material', 'hinge', 'paper_towel', 'lemon', 'jaw', 'shim', 'glass', 'sawhorse', 'refrigerator', 'shaker', 'monocle', 'work_surface', 'cellophane', 'tinsel', 'jar', 'transducer', 'cupboard', 'windowpane', 'turner', 'barrel', 'stand', 'self-propelled_vehicle', 'bird_feeder', 'lampshade', 'sweatband', 'colander', 'comb', 'caldron', 'range_hood', 'flow', 'brim', 'enamel', 'bust', 'stove', 'knob', 'wall_socket', 'thumbtack', 'embroidery', 'carafe', 'stockpot', 'buffer', 'cream_pitcher', 'headdress', 'chamber', 'palette', 'ink_cartridge', 'radiotelephone', 'connection', 'projectile', 'lamp', 'kettle', 'grandfather_clock', 'tiara', 'clamp', 'book', 'railing', 'hole', 'bow', 'weight', 'basin', 'television_camera', 'medallion', 'molding', 'apparel', 'toilet', 'stopwatch', 'flatbed', 'ski', 'autoclave', 'pouch', 'footwear', 'platter', 'dish_rack', 'dart', 'power_shovel', 'light-emitting_diode', 'passageway', 'jewelled_headdress', 'slat', 'award', 'basketball', 'telephone_receiver', 'drinking_vessel', 'cup', 'doorjamb', 'pendulum', 'eraser', 'scrub_brush', 'iron', 'furnace', 'packaging', 'sandal', 'dustpan', 'fuse', 'calculator', 'knickknack', 'bath_towel', 'plastic_wrap', 'cleaver', 'baby_buggy', 'boiler', 'scale', 'roaster', 'saucepan', 'ashcan', 'barbell', 'board_game', 'mechanical_system', 'cruet', 'crusher', 'shelter', 'audio_system', 'bookend', 'tea_bag', 'mug', 'crank', 'cone', 'tripod', 'attire', 'pepper_mill', 'stairwell', 'chopping_board', 'sports_equipment', 'headboard', 'saucer', 'lego', 'curtain', 'surgical_instrument', 'reflector', 'skirt', 'cutlery', 'orange', 'skeleton', 'tidy', 'baseball_equipment', 'squeezer', 'wrapping', 'bangle', 'power_saw', 'respirator', 'optical_disk', 'duplicator', 'swivel_chair', 'minibike', 'handset', 'edge', 'coupling', 'hand_towel', 'package', 'globe', 'skateboard', 'watch', 'figure', 'measuring_instrument', 'peg', 'computer_keyboard', 'shirt', 'monitor', 'toolbox', 'grill', 'gym_shoe', 'bedroom_furniture', 'sorter', 'videodisk', 'strip', 'hub', 'headband', 'graphic_art', 'jigsaw', 'football', 'apparatus', 'album', 'pipe', 'flight', 'accessory', 'bookcase', 'magnet', 'facility', 'webbing', 'scrapbook', 'cartridge', 'wreath', 'mirror', 'slate', 'canister', 'socket', 'notebook', 'straight_pin', 'receptacle', 'mask', 'laminate', 'guitar', 'motorboat', 'meter', 'folder', 'intake', 'rope', 'bucket', 'keyboard', 'coatrack', 'bit', 'optical_device', 'earphone', 'puppet', 'deck', 'scantling', 'chopstick', 'chisel', 'briefcase', 'bed', 'stapler', 'camera', 'drawer', 'hook', 'stairs', 'coffee_maker', 'gear', 'electronic_equipment', 'gourd', 'reproducer', 'pole', 'electric_refrigerator', 'likeness', 'kit', 'webcam', 'mural', 'printer', 'potato', 'treadmill', 'necktie', 'probe', 'backing', 'wardrobe', 'weaponry', 'lamination', 'collage', 'generator', 'bulldog_clip', 'white_goods', 'office_furniture', 'vase', 'carryall', 'weapon', 'fluorescent', 'valve', 'hygrometer', 'jamb', 'pulley', 'blind', 'windshield', 'carton', 'gown', 'sill', 'coffee_cup', 'circuit_board', 'rotating_mechanism', 'paintbrush', 'lumber', 'sunhat', 'fountain', 'sunglass', 'picture', 'component', 'trophy', 'bracket', 'cloche', 'garbage', 'lid', 'cashmere', 'seat', 'carpet_pad', 'topper', 'sequin', 'mail', 'telephone', 'portable_computer', 'blender', 'pick', 'jewel', 'diversion', 'rug', 'clipboard', 'doormat', 'toasting_fork', 'dressing_table', 'bottle', 'concave_shape', 'art', 'teacup', 'formalwear', 'easel', 'pot', 'router', 'slab', 'notch', 'dinner_jacket', 'capacitor', 'mallet', 'hairbrush', 'latch', 'odometer', 'paste-up', 'spoon', 'novel', 'clock', 'front', 'lantern', 'teaspoon', 'event', 'horn', 'frame', 'strainer', 'pendulum_clock', 'magazine', 'signaling_device', 'paperback_book', 'flower_arrangement', 'earring', 'keyboard_instrument', 'soup_ladle', 'ratchet', 'jean', 'teddy', 'doorframe', 'bottle_opener', 'board', 'neckwear', 'siren', 'balloon', 'stereo', 'goblet', 'joint'}
DUSTYABLE_OBJECT_TYPES.add('bottom_cabinet_no_top')     # somehow missed by parsing script

def get_aabb_volume(lo: Array, hi: Array) -> float:
    """Simple utility function to compute the volume of an aabb.

    lo refers to the minimum values of the bbox in the x, y and z axes,
    while hi refers to the highest values. Both lo and hi must be three-
    dimensional.
    """
    assert np.all(hi >= lo)
    dimension = hi - lo
    return dimension[0] * dimension[1] * dimension[2]


def get_closest_point_on_aabb(xyz: List, lo: Array, hi: Array) -> List[float]:
    """Get the closest point on an aabb from a particular xyz coordinate."""
    assert np.all(hi >= lo)
    closest_point_on_aabb = [0.0, 0.0, 0.0]
    for i in range(3):
        # if the coordinate is between the min and max of the aabb, then
        # use that coordinate directly
        if xyz[i] < hi[i] and xyz[i] > lo[i]:
            closest_point_on_aabb[i] = xyz[i]
        else:
            if abs(xyz[i] - hi[i]) < abs(xyz[i] - lo[i]):
                closest_point_on_aabb[i] = hi[i]
            else:
                closest_point_on_aabb[i] = lo[i]
    return closest_point_on_aabb


def get_scene_body_ids(
    env: "BehaviorEnv",
    include_self: bool = False,
    include_right_hand: bool = False,
) -> List[int]:
    """Function to return a list of body_ids for all objects in the scene for
    collision checking depending on whether navigation or grasping/ placing is
    being done."""
    ids = []
    for obj in env.scene.get_objects():
        if isinstance(obj, URDFObject):
            # We want to exclude the floor since we're always floating and
            # will never practically collide with it, but if we include it
            # in collision checking, we always seem to collide.
            if obj.name != "floors":
                ids.extend(obj.body_ids)

    if include_self:
        ids.append(env.robots[0].parts["left_hand"].get_body_id())
        ids.append(env.robots[0].parts["body"].get_body_id())
        ids.append(env.robots[0].parts["eye"].get_body_id())
        if not include_right_hand:
            ids.append(env.robots[0].parts["right_hand"].get_body_id())

    return ids


def detect_collision(bodyA: int, object_in_hand: Optional[int] = None) -> bool:
    """Detects collisions between bodyA in the scene (except for the object in
    the robot's hand)"""
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id in [bodyA, object_in_hand]:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision


def detect_robot_collision(robot: "BaseRobot") -> bool:
    """Function to detect whether the robot is currently colliding with any
    object in the scene."""
    object_in_hand = robot.parts["right_hand"].object_in_hand
    return (detect_collision(robot.parts["body"].body_id)
            or detect_collision(robot.parts["left_hand"].body_id)
            or detect_collision(robot.parts["right_hand"].body_id,
                                object_in_hand))


def reset_and_release_hand(env: "BehaviorEnv") -> None:
    """Resets the state of the right hand."""
    env.robots[0].set_position_orientation(env.robots[0].get_position(),
                                           env.robots[0].get_orientation())
    for _ in range(50):
        env.robots[0].parts["right_hand"].set_close_fraction(0)
        env.robots[0].parts["right_hand"].trigger_fraction = 0
        p.stepSimulation()


def get_delta_low_level_base_action(robot_z: float,
                                    original_orientation: Tuple,
                                    old_xytheta: Array, new_xytheta: Array,
                                    action_space_shape: Tuple) -> Array:
    """Given a base movement plan that is a series of waypoints in world-frame
    position space, convert pairs of these points to a base movement action in
    velocity space.

    Note that we cannot simply subtract subsequent positions because the
    velocity action space used by BEHAVIOR is not defined in the world
    frame, but rather in the frame of the previous position.
    """
    ret_action = np.zeros(action_space_shape, dtype=np.float32)

    # First, get the old and new position and orientation in the world
    # frame as numpy arrays
    old_pos = np.array([old_xytheta[0], old_xytheta[1], robot_z])
    old_orn_quat = p.getQuaternionFromEuler(
        np.array(
            [original_orientation[0], original_orientation[1],
             old_xytheta[2]]))
    new_pos = np.array([new_xytheta[0], new_xytheta[1], robot_z])
    new_orn_quat = p.getQuaternionFromEuler(
        np.array(
            [original_orientation[0], original_orientation[1],
             new_xytheta[2]]))

    # Then, simply get the delta position and orientation by multiplying the
    # inverse of the old pose by the new pose
    inverted_old_pos, inverted_old_orn_quat = p.invertTransform(
        old_pos, old_orn_quat)
    delta_pos, delta_orn_quat = p.multiplyTransforms(inverted_old_pos,
                                                     inverted_old_orn_quat,
                                                     new_pos, new_orn_quat)

    # Finally, convert the orientation back to euler angles from a quaternion
    delta_orn = p.getEulerFromQuaternion(delta_orn_quat)

    ret_action[0:3] = np.array([delta_pos[0], delta_pos[1], delta_orn[2]])

    return ret_action


def get_delta_low_level_hand_action(
    body: "BRBody",
    old_pos: Union[Sequence[float], Array],
    old_orn: Union[Sequence[float], Array],
    new_pos: Union[Sequence[float], Array],
    new_orn: Union[Sequence[float], Array],
) -> Array:
    """Given a hand movement plan that is a series of waypoints for the hand in
    position space, convert pairs of these points to a hand movement action in
    velocity space.

    Note that we cannot simply subtract subsequent positions because the
    velocity action space used by BEHAVIOR is not defined in the world
    frame, but rather in the frame of the previous position.
    """
    # First, convert the supplied orientations to quaternions
    old_orn = p.getQuaternionFromEuler(old_orn)
    new_orn = p.getQuaternionFromEuler(new_orn)

    # Next, find the inverted position of the body (which we know shouldn't
    # change, since our actions move either the body or the hand, but not
    # both simultaneously)
    inverted_body_new_pos, inverted_body_new_orn = p.invertTransform(
        body.new_pos, body.new_orn)
    # Use this to compute the new pose of the hand w.r.t the body frame
    new_local_pos, new_local_orn = p.multiplyTransforms(
        inverted_body_new_pos, inverted_body_new_orn, new_pos, new_orn)

    # Next, compute the old pose of the hand w.r.t the body frame
    inverted_body_old_pos = inverted_body_new_pos
    inverted_body_old_orn = inverted_body_new_orn
    old_local_pos, old_local_orn = p.multiplyTransforms(
        inverted_body_old_pos, inverted_body_old_orn, old_pos, old_orn)

    # The delta position is simply given by the difference between these
    # positions
    delta_pos = np.array(new_local_pos) - np.array(old_local_pos)

    # Finally, compute the delta orientation
    inverted_old_local_orn_pos, inverted_old_local_orn_orn = p.invertTransform(
        [0, 0, 0], old_local_orn)
    _, delta_orn = p.multiplyTransforms(
        [0, 0, 0],
        new_local_orn,
        inverted_old_local_orn_pos,
        inverted_old_local_orn_orn,
    )

    delta_trig_frac = 0
    action = np.concatenate(
        [
            np.zeros((10), dtype=np.float32),
            np.array(delta_pos, dtype=np.float32),
            np.array(p.getEulerFromQuaternion(delta_orn), dtype=np.float32),
            np.array([delta_trig_frac], dtype=np.float32),
        ],
        axis=0,
    )

    return action


def check_nav_end_pose(
        env: "BehaviorEnv", obj: Union["URDFObject", "RoomFloor"],
        pos_offset: Array) -> Optional[Tuple[List[int], List[int]]]:
    """Check that the robot can reach pos_offset from the obj without (1) being
    in collision with anything, or (2) being blocked from obj by some other
    solid object.

    If this is true, return the ((x,y,z),(roll, pitch, yaw)), else
    return None
    """
    valid_position = None
    state = p.saveState()
    obj_pos = obj.get_position()
    pos = [
        pos_offset[0] + obj_pos[0],
        pos_offset[1] + obj_pos[1],
        env.robots[0].initial_z_offset,
    ]
    yaw_angle = np.arctan2(pos_offset[1], pos_offset[0]) - np.pi
    orn = [0, 0, yaw_angle]
    env.robots[0].set_position_orientation(pos, p.getQuaternionFromEuler(orn))
    eye_pos = env.robots[0].parts["eye"].get_position()
    ray_test_res = p.rayTest(eye_pos, obj_pos)
    # Test to see if the robot is obstructed by some object, but make sure
    # that object is not either the robot's body or the object we want to
    # pick up!
    blocked = len(ray_test_res) > 0 and (ray_test_res[0][0] not in (
        env.robots[0].parts["body"].get_body_id(),
        obj.get_body_id(),
    ))
    if not detect_robot_collision(env.robots[0]) and not blocked:
        valid_position = (pos, orn)

    p.restoreState(state)
    p.removeState(state)

    return valid_position


def check_hand_end_pose(env: "BehaviorEnv", obj: Union["URDFObject",
                                                       "RoomFloor"],
                        pos_offset: Array) -> bool:
    """Check that the robot's hand can reach pos_offset from the obj without
    being in collision with anything.

    If this is true, return True, else return False.
    """
    ret_bool = False
    state = p.saveState()
    obj_pos = obj.get_position()
    hand_pos = (
        pos_offset[0] + obj_pos[0],
        pos_offset[1] + obj_pos[1],
        pos_offset[2] + obj_pos[2],
    )
    env.robots[0].parts["right_hand"].set_position(hand_pos)
    if not detect_robot_collision(env.robots[0]):
        ret_bool = True

    p.restoreState(state)
    p.removeState(state)

    return ret_bool


def load_checkpoint_state(s: State,
                          env: "BehaviorEnv",
                          reset: bool = False) -> None:
    """Sets the underlying iGibson environment to a particular saved state.

    When reset is True we will create a new BehaviorEnv and load our
    checkpoint into it. This will ensure that all the information from
    previous environment steps are reset as well.
    """
    assert s.simulator_state is not None
    # Get the new_task_num_task_instance_id associated with this state
    # from s.simulator_state.
    new_task_num_task_instance_id = (int(s.simulator_state.split("-")[0]),
                                     int(s.simulator_state.split("-")[1]))
    # If the new_task_num_task_instance_id is new, then we need to load
    # a new iGibson behavior env with our random seed saved in
    # env.new_task_num_task_instance_id_to_igibson_seed. Otherwise
    # we're already in the correct environment and can just load the
    # checkpoint. Also note that we overwrite the task.init saved checkpoint
    # so that it's compatible with the new environment!
    env.task_num = new_task_num_task_instance_id[0]
    # Since demo trajectories seeds are not saved, a seed is generated here if
    # one does not exist yet for the task num and task instance id pair.
    if not new_task_num_task_instance_id in \
        env.task_num_task_instance_id_to_igibson_seed:
        env.task_num_task_instance_id_to_igibson_seed[
            new_task_num_task_instance_id] = 0
    if (new_task_num_task_instance_id != (env.task_num, env.task_instance_id)
            and CFG.behavior_randomize_init_state) or reset:
        env.task_instance_id = new_task_num_task_instance_id[1]
        # Frame count is overwritten by set_igibson_behavior_env and needs to
        # be preserved across resets. So we save it before and set it after
        # we reset the env.
        frame_count = env.igibson_behavior_env.simulator.frame_count
        env.set_igibson_behavior_env(
            task_num=env.task_num,
            task_instance_id=new_task_num_task_instance_id[1],
            seed=env.task_num_task_instance_id_to_igibson_seed[
                new_task_num_task_instance_id])
        env.igibson_behavior_env.simulator.frame_count = frame_count
        env.set_options()
        env.current_ig_state_to_state(
        )  # overwrite the old task_init checkpoint file!
        env.igibson_behavior_env.reset()
    behavior_task_name = CFG.behavior_task_list[0] if len(
        CFG.behavior_task_list) == 1 else "all"
    checkpoint_file_str = (
        f"tmp_behavior_states/{CFG.behavior_scene_name}__" +
        f"{behavior_task_name}__{CFG.num_train_tasks}__" +
        f"{CFG.seed}__{env.task_num}__{env.task_instance_id}")
    frame_num = int(s.simulator_state.split("-")[2])
    try:
        load_checkpoint(env.igibson_behavior_env.simulator,
                        checkpoint_file_str, frame_num)
    except p.error as _:
        print(f"tmp_behavior_states_dir: {os.listdir(checkpoint_file_str)}")
        raise ValueError(
            f"Could not load pybullet state for {checkpoint_file_str}, " +
            f"frame {frame_num}")

    np.random.seed(env.task_num_task_instance_id_to_igibson_seed[
        new_task_num_task_instance_id])
    # We step the environment to update the visuals of where the robot is!
    env.igibson_behavior_env.step(
        np.zeros(env.igibson_behavior_env.action_space.shape))


def create_ground_atom_dataset_behavior(
        trajectories: Sequence[LowLevelTrajectory], predicates: Set[Predicate],
        env: "BehaviorEnv") -> List[GroundAtomTrajectory]:  # pragma: no cover
    """Apply all predicates to all trajectories in the dataset."""
    ground_atom_dataset = []
    for traj in trajectories:
        atoms = []
        for s in traj.states:
            # If th environment is BEHAVIOR we need to load the state before
            # we call the predicate classifiers.
            load_checkpoint_state(s, env)
            atoms.append(abstract(s, predicates))
        ground_atom_dataset.append((traj, atoms))
    return ground_atom_dataset
