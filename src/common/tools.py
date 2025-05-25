# ================================================
# Prepare Torch
# ================================================
import yaml
import unicodedata
import string
import torch

device = torch.device("cuda"
                      if torch.cuda.is_available()
                      else "cpu")

torch.set_default_device(device)
# print(f"using {str(torch.get_default_device()).upper()}.")

# ================================================
# Prepare the Data
# ================================================

# "_" represents an out-of-vocabulary character or
# any character we are not handling in our model
allowed_characters = string.ascii_letters + " .,;'-"
n_letters = len(allowed_characters)

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )

# print(f"converted 'máquina' to {unicode_to_ascii('máquina')}.")

# ================================================
# Turning Names into Tensors
# ================================================
# Find letter index from all_letters, e.g., "a" = 0,
# "b" = 1, etc.


def letter_to_index(letter):
    if letter not in allowed_characters:
        return allowed_characters.find('_')
    else:
        return allowed_characters.find(letter)

# Turn a line into a <line_length x 1 x n_letters> tensor


def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor
# print(f"the name 'Ahn' becomes {line_to_tensor('Ahn')}");


# ================================================
# Miscellaneous
# ================================================


def load_config():
    # Read in the config file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config
