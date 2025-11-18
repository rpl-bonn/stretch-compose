# object_priors.py
import random

# 60 candidate objects
OBJECTS = [
    "apartment_keys", "mobile_phone", "spectacles", "watch", "wallet",
    "laptop", "tablet", "remote_control", "headphones", "charger",
    "backpack", "jacket", "scarf", "tshirt", "jeans",
    "socks", "shoes", "umbrella", "book", "notebook",
    "pen", "folder", "board_game", "card_game", "magazine",
    "mug", "cup", "plate", "bottle", "water_jug",
    "spoon", "fork", "knife", "pan", "saucepan",
    "hairbrush", "perfume", "makeup_box", "pill_box", "medicine_bottle",
    "toy_car", "teddy_bear", "lego_box", "ball", "tennis_ball",
    "flashlight", "power_drill", "scissors", "screwdriver", "hammer",
    "blanket", "pillow", "bedsheet", "towel", "yoga_mat",
    "camera", "tripod", "game_controller", "earbuds", "keys_backup"
]

# Semantic priors: plausible furniture for each object
SEMANTIC_PRIORS = {
    # Personal items
    "apartment_keys": ["round_table", "shoe_rack", "nightstand_left"],
    "keys_backup": ["wardrobe", "dressing_table"],
    "mobile_phone": ["nightstand_right", "sofa", "bed"],
    "spectacles": ["nightstand_left", "dressing_table"],
    "watch": ["nightstand_left", "dressing_table"],
    "wallet": ["round_table", "nightstand_right"],

    # Electronics
    "laptop": ["couchtisch", "small_table", "bed"],
    "tablet": ["couchtisch", "bed", "sofa"],
    "remote_control": ["couchtisch", "sofa"],
    "headphones": ["couchtisch", "bed", "nightstand_right"],
    "charger": ["nightstand_left", "dressing_table", "small_table"],

    # Clothing / accessories
    "backpack": ["wardrobe", "small_table", "sofa"],
    "jacket": ["wardrobe", "chair", "sofa"],
    "scarf": ["wardrobe", "dressing_table"],
    "tshirt": ["wardrobe", "bed"],
    "jeans": ["wardrobe", "bed"],
    "socks": ["wardrobe", "nightstand_left"],
    "shoes": ["shoe_rack", "wardrobe"],
    "umbrella": ["shoe_rack", "wardrobe"],

    # Reading / office
    "book": ["bookshelf_kallax", "couchtisch", "bed"],
    "notebook": ["dressing_table", "small_table", "couchtisch"],
    "pen": ["dressing_table", "small_table"],
    "folder": ["bookshelf_kallax", "dressing_table"],
    "magazine": ["sofa", "couchtisch", "round_table"],

    # Games / leisure
    "board_game": ["bookshelf_kallax", "couchtisch"],
    "card_game": ["bookshelf_kallax", "couchtisch"],
    "toy_car": ["bookshelf_kallax", "dressing_table"],
    "teddy_bear": ["bed", "bookshelf_kallax"],
    "lego_box": ["bookshelf_kallax", "wardrobe"],
    "ball": ["bookshelf_kallax", "shoe_rack"],
    "tennis_ball": ["bookshelf_kallax", "shoe_rack"],

    # Kitchenware
    "mug": ["round_table", "couchtisch", "bookshelf_kallax"],
    "cup": ["round_table", "bookshelf_kallax"],
    "plate": ["round_table", "wardrobe"],  # misplaced option too
    "bottle": ["couchtisch", "round_table"],
    "water_jug": ["round_table", "bookshelf_kallax"],
    "spoon": ["couchtisch", "round_table"],
    "fork": ["couchtisch", "round_table"],
    "knife": ["couchtisch", "round_table"],
    "pan": ["wardrobe", "bookshelf_kallax"],  # unusual
    "saucepan": ["wardrobe", "bookshelf_kallax"],

    # Toiletries / health
    "hairbrush": ["dressing_table", "nightstand_right"],
    "perfume": ["dressing_table", "nightstand_left"],
    "makeup_box": ["dressing_table", "nightstand_right"],
    "pill_box": ["nightstand_left", "dressing_table"],
    "medicine_bottle": ["nightstand_right", "dressing_table"],

    # Tools
    "flashlight": ["nightstand_left", "shoe_rack"],
    "power_drill": ["wardrobe", "bookshelf_kallax"],
    "scissors": ["dressing_table", "small_table"],
    "screwdriver": ["wardrobe", "bookshelf_kallax"],
    "hammer": ["wardrobe", "bookshelf_kallax"],

    # Bedding
    "blanket": ["bed", "wardrobe"],
    "pillow": ["bed", "wardrobe"],
    "bedsheet": ["bed", "wardrobe"],
    "towel": ["wardrobe", "dressing_table"],
    "yoga_mat": ["wardrobe", "small_table"],

    # Media / gadgets
    "camera": ["dressing_table", "wardrobe"],
    "tripod": ["wardrobe", "small_table"],
    "game_controller": ["couchtisch", "tv_stand"],
    "earbuds": ["nightstand_right", "dressing_table"]
}

def sample_objects(num_objects: int = None):
    """
    Sample a subset of objects for a household (40–50).
    """
    if num_objects is None:
        num_objects = random.randint(40, 50)
    return random.sample(OBJECTS, num_objects)
