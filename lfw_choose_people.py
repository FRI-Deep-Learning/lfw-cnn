import os

THRESHOLD = 20

names_file = open("lfw-names.txt", "r")

results = []

for line in names_file:
    parts = line[:-1].split("\t")
    
    if int(parts[1]) >= THRESHOLD:
        print("Accepting", parts[0])

        results.append((parts[0], parts[1]))


print()
print("== TOTAL PEOPLE:", len(results))

total_images = sum([int(p[1]) for p in results])

print("== TOTAL IMAGES:", total_images)

accepted_people_file = open("lfw-names-accepted.txt", "w")

accepted_people_file.write("\n".join([p[0] + "\t" + p[1] for p in results]))