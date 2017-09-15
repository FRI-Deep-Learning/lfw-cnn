import os.path
import numpy as np
from tqdm import tqdm
import sys
import random
import cv2

names_file = open("lfw-names-accepted.txt", "r")

people = []

for line in names_file:
    parts = line.split("\t")
    people.append((parts[0], int(parts[1])))

names_file.close()

input_dir = "faces"

def images_for_person(person):
    images = []

    for img_idx in range(1, person[1] + 1):
        images.append(os.path.join(input_dir, person[0] + "_" + str(img_idx).zfill(4) + ".pgm"))
        
    return images

def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    assert pgmf.readline() == b'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster

def read_image(image):
    f = open(image, "rb")
    image_pgm = read_pgm(f)
    f.close()
    return np.array(image_pgm)


def pickle(num_pairs, outfile_prefix):

    print("> Choosing same-person pairs...")

    same_person_pairs_names = []

    p_idx = 0
    for i in tqdm(range(0, num_pairs)):
        images = images_for_person(people[p_idx])

        image1 = random.choice(images)
        image2 = random.choice(images)

        same_person_pairs_names.append((people[p_idx][0], image1, image2)) # Append a tuple like ("Name", "Image 1 Name", "Image 2 Name")

        p_idx += 1

        if p_idx == len(people):
            p_idx = 0
    
    print("> Choosing different-people pairs...")

    different_people_pairs_names = []

    p_idx = 0
    for i in tqdm(range(0, num_pairs)):
        person = people[p_idx]
        other_person = random.choice([p for p in people if p[0] != person]) # Choose a random person that is not the current person.

        person_images = images_for_person(person)
        other_person_images = images_for_person(other_person)

        image1 = random.choice(person_images)
        image2 = random.choice(other_person_images)

        different_people_pairs_names.append((person[0], image1, other_person[0], image2)) # Append a tuple like ("Person 1", "Image for Person 1", "Person 2", "Image for person 2")

        p_idx += 1

        if p_idx == len(people):
            p_idx = 0

    print("> Pickling same-person-pairs...")

    same_person_pairs_x = []

    for same_person_pair_tuple in tqdm(same_person_pairs_names):
        person = same_person_pair_tuple[0]
        image1_name = same_person_pair_tuple[1]
        image2_name = same_person_pair_tuple[2]

        image1 = read_image(image1_name)
        image2 = read_image(image2_name)

        both_images = np.stack((image1, image2), axis = 2)
        same_person_pairs_x.append(both_images)


    print("> Pickling different-people-pairs...")

    different_people_pairs_x = []

    for different_people_pair_tuple in tqdm(different_people_pairs_names):
        person1 = different_people_pair_tuple[0]
        image1_name = different_people_pair_tuple[1]

        person2 = different_people_pair_tuple[2]
        image2_name = different_people_pair_tuple[3]

        image1 = read_image(image1_name)
        image2 = read_image(image2_name)

        both_images = np.stack((image1, image2), axis = 2)
        different_people_pairs_x.append(both_images)


    same_person_pairs_y = np.ones((num_pairs), dtype=np.uint8)
    different_people_pairs_y = np.zeros((num_pairs), dtype=np.uint8)

    """
        Combine the same_person_pairs_x and different_people_pairs_x and do it for the y's as well.
        This way, we have one input array and one output array. Half of the inputs are same-person
        and half are different-people, in the same way that half of the outputs are 1's and half are 0's.
    """

    pairs_x = np.append(np.array(same_person_pairs_x), np.array(different_people_pairs_x), axis = 0)
    pairs_y = np.append(same_person_pairs_y, different_people_pairs_y, axis = 0)

    """
        Finally, pickle these arrays into two files, "pairs_x.pickle" and "pairs_y.pickle".
    """

    print("> Shapes:")
    print("\t", pairs_x.shape)
    print("\t", pairs_y.shape)

    print("> Saving to file...")

    np.save(outfile_prefix + "_pairs_x", pairs_x)
    np.save(outfile_prefix + "_pairs_y", pairs_y)

if len(sys.argv) < 3:
    print("Arguments: <num training pairs> <num testing pairs>")
    exit(1)

num_training_pairs = int(sys.argv[1])
num_testing_pairs = int(sys.argv[2])

print("--- Pickling training set ---")
pickle(num_training_pairs, "train")

print("--- Pickling testing set ---")
pickle(num_testing_pairs, "test")