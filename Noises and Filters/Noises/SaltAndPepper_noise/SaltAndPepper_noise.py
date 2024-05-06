import cv2
import numpy as np

img = cv2.imread("../../image2.jpeg", cv2.IMREAD_GRAYSCALE)

salt_prob = 0.01  
pepper_prob = 0.01  

noisy_image = np.copy(img)

salt_mask = np.random.rand(*img.shape[:2]) < salt_prob
noisy_image[salt_mask] = 255

    
pepper_mask = np.random.rand(*img.shape[:2]) < pepper_prob
noisy_image[pepper_mask] = 0

cv2.imwrite('image_SaltAndPepper_noise.jpeg', noisy_image)


cv2.imshow("Original Image", img)
cv2.imshow("Noisy Image with Salt and Pepper Noise", noisy_image)


#Press q or esc to exit;
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break


cv2.destroyAllWindows()
