from pylab import *
img = imread('COM6115/Lab_6/images/chick.png')
imshow(img)
print(img.shape)
(rows, cols, d3) = img.shape
print(cols)
print(img[0, 0, 0])
print(img[0, 0])
#show()



for i in range (rows):
    for j in range (cols):
        for k in range(d3):
            img[i, j, k] = 1 - img[i, j, k]

print(img.shape)
print(img[0, 0, 0])
print(img[0, 0])



for i in range (rows):
    for j in range(cols):
        pixel1 = img[i, j]
        if sum(pixel1) < 1.5:
            img[i, j] = (.0, .0, .0)

print(img.shape)
print(img[0, 0, 0])
print(img[0, 0])
show()
