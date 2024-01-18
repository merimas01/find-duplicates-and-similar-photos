import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import imghdr

# get the file dialog for choosing a file
Tk().withdraw()

# show an "Open" dialog box and return the path to the selected file
selected_file = askopenfilename(title="Select an image")

# get the directory of selected path of image
directory = os.path.dirname(selected_file)

# split the name of the selected image
sf_name = os.path.split(selected_file)
sf_name = sf_name[1]

# os.walk generates the file names in the directory so we can loop through them
walker = os.walk(directory)

# ORB algorithm
orb = cv2.ORB_create(nfeatures=1000)

# get key points and descriptors
kp1, des1 = orb.detectAndCompute(cv2.imread(selected_file), None)

# number of calculated keypoints
kp1_number = len(kp1)

# array of good points
good = []

# counters
counter_similar_photos = 0
counter_duplicate_photos = 0

# array of similar images
similar_images = dict()


def match_images(img):
    # get keypoints and descriptors
    kp, des = orb.detectAndCompute(img, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # match the descriptors using knnMatch
    matches = bf.knnMatch(des1, des, k=2)

    # find the good matches
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return len(kp)


# loop through the folders, sub folders and files in given directory
for folder, sub_folder, files in walker:
    for file in files:
        # get the filepath of current file
        filepath = os.path.join(folder, file)

        # check if a file is an image
        extension = imghdr.what(filepath)

        if extension == "jpeg" or extension == "jpg" or extension == "png":
            # get the name of the file
            name = os.path.split(filepath)
            name = name[1]

            # read a file
            img = cv2.imread(filepath)

            if (name != sf_name and folder == directory) or (folder != directory):
                kp_number = match_images(img)
                # print(name, len(good), kp_number, kp1_number)

                if len(good) == kp_number and kp_number == kp1_number: 
                   # images are duplicates
                    counter_duplicate_photos += 1
                    os.remove(filepath)
                elif (
                    len(good) > 20
                    and len(good) < 1000
                    and (kp_number != kp1_number or kp_number == kp1_number)
                ):     
                     # images are similar
                    counter_similar_photos += 1
                    similar_images[filepath] = img
        # refresh the value
        good = []

print(f"{counter_similar_photos} similar images detected!")
print(f"{counter_duplicate_photos} duplicate images removed!")

i = 0
for key, image in similar_images.items():
    cv2.imshow(f"similar_image_{i}: {key}", image)
    print(f"similar_image_{i}: {key}")
    cv2.waitKey(0)
    i += 1
