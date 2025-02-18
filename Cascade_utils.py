import os


def generate_negative_description_file():
    with open('data/neg.txt', 'w') as f:
        for filename in os.listdir('pictures/negative'):
            f.write('pictures/negative/' + filename + '\n')

def generate_positive_description_file():
    with open('data/pos.txt', 'w') as f:
        for filename in os.listdir('pictures/positive'):
            f.write('pictures/positive/' + filename + '\n')


generate_negative_description_file()
generate_positive_description_file()