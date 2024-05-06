import cv2
import numpy as np
from matplotlib import pyplot as plt

# read as a gray scale image

img = cv2.imread("../image.jpeg", 0)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()


img_transformed = np.fft.fft2(img) # high frequency in center, so we have to shift it.
plt.imshow(np.log1p(np.abs(img_transformed)), cmap = 'grey')
plt.title("Image in Frequency Domain but high frequencies in center")
plt.axis("off")
plt.show()

img_transformed_shifted = np.fft.fftshift(img_transformed) 
plt.imshow(np.log1p(np.abs(img_transformed_shifted)), cmap = 'grey')
plt.title("Image in Frequency Domain shifted so, lower frequencies in center\high pass filtering Image spectrum")
plt.axis("off")
plt.show()


#perfrom low pass filter
M, N = img.shape
high_pass_mask = np.ones((M,N), dtype=np.float32)
D0 = 30

for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        if D <= D0:
            high_pass_mask[u, v] = 0
        else:
            high_pass_mask[u, v] = 1



plt.imshow(high_pass_mask, cmap="gray") # high-pass filter
plt.title("high Pass Filter")
plt.axis("off")
plt.show()


high_pass_filtered_image = img_transformed_shifted * high_pass_mask
plt.imshow(np.log1p(np.abs(high_pass_filtered_image)), cmap = 'grey')
plt.title("image in Frequehigh Domain after applying the high Pass Filter")
plt.show()           


img_transformed = np.fft.ifftshift(high_pass_filtered_image)
plt.imshow(np.log1p(np.abs(img_transformed)), cmap = 'grey')
plt.title("image in Frequency Domain after applying the high Pass \n Filter but low frequencies NOT in center")
plt.show()   

img = np.abs(np.fft.ifft2(img_transformed))
plt.imshow(img, cmap="gray")
plt.title("image in Spatial Domain after applying the High Pass Filter")
plt.axis("off")
plt.show()

