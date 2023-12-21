import os

def save_list(path, list_values):
    with open(path, 'w+') as file:
        for val in list_values:
            file.write('%s\n' %val)