# Below script is used to convert color image to grayscale.
# Steps: 1. Convert RGB to grayscale
#  	 2. Get the pixel matrix
#	 3. Load the pixel matrix to pickle file/simple text file
#	 4. Use this pickle file to run CPU and GPU convolutions.


import cv2
import matplotlib.image as img
import numpy as np

# Step-1: Conversion of image to Greyscale
img_name = 'naveen.jpg'
grayscale_img = 'naveen_gray_test.jpg'

image = cv2.imread(img_name)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Print the original image and grayscale image
cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)

# Save the grayscale image to the disk
#cv2.imwrite(grayscale_img, gray)


# Step-2: Get the pixel matrix and load it
# Now load the grayscale image and get the 2D pixel matrix
#gray_img = cv2.imread('grayscale_img')
gray_img = img.imread(grayscale_img)
print(np.asmatrix(gray_img))


### Tried to write the resulting numpy array to output file
outfile = 'nav'

#with open('file_nav.txt','w') as of:
#np.savetxt("test2.txt", gray_img, fmt="%2.3f", delimiter=",")
#np.ndarray.tofile(outfile, sep="", format="%s")
# Save the matrix to an appropriate file
#with open(outfile, 'w') as of:
#for data in gray_img:
#np.savetxt(outfile, gray_img)
#of.close()

#file = open("nav",mode='r')
#matrix_GS_img = file.read()
#file_read = 'nav.npy'
#matr = np.load(file_read)
#print(matrix_GS_img)
#file.close()


#np_arr = np.array([[2,2],[1,1]])
#print(np_arr)

# Step-3:The below code is used to write a 2D array to output_file.
with open("naveen.txt","w") as wFile:
    wFile.write("[")
    for i in range(gray_img.shape[0]):
        wFile.write("[")
        for j in range(gray_img.shape[1]):
            wFile.write(str(gray_img[i][j]))
            if j!=gray_img.shape[1]-1:
                wFile.write(", ")
        if i!=gray_img.shape[0]-1:
            wFile.write("],\n")
        else:
            wFile.write("]\n")
    wFile.write("]")
wFile.close()

'''
with open("naveen.txt","w") as wFile:
    wFile.write("[")
    for i in range(np_arr.shape[0]):
        wFile.write("[")
        for j in range(np_arr.shape[1]):
            wFile.write(str(np_arr[i][j]))
            if j!=np_arr.shape[1]-1:
                wFile.write(", ")
        if i!=np_arr.shape[0]-1:
            wFile.write("],\n")
        else:
            wFile.write("]\n")
    wFile.write("]")
wFile.close()
''' 
