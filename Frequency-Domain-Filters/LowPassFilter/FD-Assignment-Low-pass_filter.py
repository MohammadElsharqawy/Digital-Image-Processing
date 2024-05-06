import cv2
import numpy as np
from matplotlib import pyplot as plt

# read as a gray scale image
img = cv2.imread("image.jpeg", 0)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

img_transformed = np.fft.fft2(img) # high frequency in center, so we have to shift it.
plt.imshow(np.log1p(np.abs(img_transformed)), cmap = 'grey')
plt.title("Image in Frequency Domain but high frequencies in center")
plt.show()

img_transformed_shifted = np.fft.fftshift(img_transformed) 
plt.imshow(np.log1p(np.abs(img_transformed_shifted)), cmap = 'grey')
plt.title("Image in Frequency Domain shifted so, lower frequencies in center\nLow pass filtering Image spectrum")
plt.show()


#perfrom low pass filter
M, N = img.shape
mask = np.zeros((M,N), dtype=np.float32)
D0 = 15

for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        if D <= D0:
            mask[u, v] = 1
        else:
            mask[u, v] = 0

plt.imshow(mask, cmap="gray") # low-pass filter
plt.title("Low Pass Filter")
plt.show()


low_pass_filtered_image = img_transformed_shifted * mask
plt.imshow(np.log1p(np.abs(low_pass_filtered_image)), cmap = 'grey')
plt.title("image in Frequency Domain after applying the Low Pass Filter")
plt.show()           


img_transformed = np.fft.ifftshift(low_pass_filtered_image)
plt.imshow(np.log1p(np.abs(img_transformed)), cmap = 'grey')
plt.title("image in Frequency Domain after applying the Low Pass \n Filter but low frequencies NOT in center")
plt.show()   

img = np.abs(np.fft.ifft2(img_transformed))
plt.imshow(img, cmap="gray")
plt.title("image in Spatial Domain after applying the Low Pass Filter")
plt.show()

