import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            file_names.append(str.split(filename,'.')[0])
            #img = cv2.resize(img , (800,1000),interpolation=cv2.INTER_AREA)
            images.append(img)
    return images

file_names = []
# Path to the folder containing multiple face images
folder_path = './Training'

# Load images from the folder
face_images = load_images_from_folder(folder_path)
print(file_names)

# Load the test image
test_img = cv2.imread('./Test/TestORG.jpg')
#test_img = cv2.resize(test_img , (800,1000),interpolation=cv2.INTER_AREA)
""" test_img = cv2.imread('./Test/TestORG.jpg')
    test_img = cv2.imread('./Test/TestAPL.jpg')
    test_img = cv2.imread('./Test/TestBNA.jpg')
    test_img = cv2.imread('./Test/TestGRP.jpg')
    test_img = cv2.imread('./Test/TestGR2.jpg') """

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors for the test image
kp1, des1 = sift.detectAndCompute(test_img, None)

# Create BFMatcher object
bf = cv2.BFMatcher()

best_match_score = 0
best_match_index = -1

des = []
test=[]
# Find keypoints and descriptors for the resized test image
kp1, des1 = sift.detectAndCompute(test_img, None)

# Iterate through each image in the folder to find the best match
for i, face_img in enumerate(face_images):

    # Find keypoints and descriptors for the face image in the folder
    kp2, des2 = sift.detectAndCompute(face_img, None)
    des.append(des2)
    # Match descriptors using knn
    matches = bf.knnMatch(des1, des2, k=2)
    test.append(matches)
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
    
    # Classify matches by length of good matches
    match_score = len(good_matches)
    print(match_score)
    
    # Update best match if the current match is better
    if match_score >= best_match_score:
        best_match_score = match_score
        best_match_index = i
        print("New best match found in the folder at index:", best_match_index)

# Display the best matching image
if best_match_index != -1:
    best_match_img = face_images[best_match_index]
    
    matches = bf.knnMatch(des1, des[best_match_index], k=2)
    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    img3 = cv2.drawMatchesKnn(test_img, kp1, best_match_img, kp2, good_matches, None, flags=2)

    img_class = str(file_names[best_match_index]).split('_')[0]

    cv2.putText(img3,img_class,(10,150),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
    
    img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img3_rgb), plt.show()
    print("Best match found in the folder at index:", best_match_index)
else:
    print("No match found in the folder.")

"""
I1 -> P10 ==> Buffer

Cycle | Frames reçues                                                           | Décodeur (thread)             | Présentation (thread)
----- | -------------                                                           | -----------------             | ---------------------
1     | I1                                                                      |                               |
2     | I1 P2                                                                   |                               |
3     | I1 P2 P3                                                                |                               |
4     | I1 P2 P3 P4                                                             |                               |
5     | I1 P2 P3 P4 I5                                                          |                               |
6     | I1 P2 P3 P4 I5 P6                                                       |                               | 
7     | I1 P2 P3 P4 I5 P6 P7                                                    |                               | 
8     | I1 P2 P3 P4 I5 P6 P7 P8                                                 |                               | 
9     | I1 P2 P3 P4 I5 P6 P7 P8 I9                                              |                               | 
10    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10                                          |                               | 
11    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10 P11                                      | I1                            |
12    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10 P11 P12                                  | I1 P2                         | I1 
13    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10 P11 P12 I13                              | I1 P2 P3                      | I1 P2 
14    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10 P11 P12 I13 P14                          | I1 P2 P3 P4                   | I1 P2 P3 
15    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10 P11 P12 I13 P14 P15                      | I1 P2 P3 P4 I5                | I1 P2 P3 P4 
16    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10 P11 P12 I13 P14 P15 P16                  | I1 P2 P3 P4 I5 P6             | I1 P2 P3 P4 I5
17    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10 P11 P12 I13 P14 P15 P16 I17              | I1 P2 P3 P4 I5 P6 P7          | I1 P2 P3 P4 I5 P6 
18    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10 P11 P12 I13 P14 P15 P16 I17 P18          | I1 P2 P3 P4 I5 P6 P7 P8       | I1 P2 P3 P4 I5 P6 P7 
19    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10 P11 P12 I13 P14 P15 P16 I17 P18 P19      | I1 P2 P3 P4 I5 P6 P7 P8 I9    | I1 P2 P3 P4 I5 P6 P7 P8 
20    | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10 P11 P12 I13 P14 P15 P16 I17 P18 P19 P20  | I1 P2 P3 P4 I5 P6 P7 P8 I9 P10| I1 P2 P3 P4 I5 P6 P7 P8 I9 


 """