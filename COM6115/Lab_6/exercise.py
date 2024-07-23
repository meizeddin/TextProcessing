from pylab import *

img = imread("COM6115/Lab_6/images/che.png")
imshow(img)
show()
#print(img.shape)
(rows, col, d3) = img.shape
#print(img[0, 0, 0])
#print(img[0, 0])


img1 = array(img)

for i in range(rows):
    for j in range(col):
        for k in range(d3):
            if img1[i, j, k] < 0.5:
                img1[i, j, k] = 0.0
            if img1[i, j, k] > 0.5:
                img1[i, j, k] = 1.0
print(img1[0, 0, 0])
print(img1[0, 0])

imshow(img1)
#show()

img2 = array(img1)

for i in range(rows):
    for j in range(col):
        pixels = img2[i, j]
        if sum(pixels) == 3.0:
            img2[i, j] = (1.0, .0, .0)

imshow(img2)
show()

img3 = array(img2)

for i in range(50, 170):
    for j in range(50, 150):
        pixels = img3[i, j]
        if sum(pixels) == 1.0:
            img3[i, j] = (1.0, 1.0, 1.0)

imshow(img3)
#show()

img4 = array(img)

for i in range(rows):
    for j in range(col):
        pixels = img4[i, j]
        if sum(pixels) > 0.66:
            img4[i, j] = (1.0, 0.0, 0.0)
        if sum(pixels) < 0.33:
            img4[i, j] = (0.0, 1.0, 0.0)
        if  0.33 < sum(pixels) < 0.66:
            img4[i, j] = (0.0, 0.0, 1.0)

imshow(img4)
#show()

img5 = imread("COM6115/Lab_6/images/chick.png")
(row, col, d) = img5.shape
img6 = array(img)
(rows, cols, d3) = img6.shape
for i in range(0, 99):
    for j in range(0, 70):
        img6[i+60, j+50] = img5[i, j]

imshow(img6)
show()

