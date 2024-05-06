import cv2
import numpy as np

img = cv2.imread("../../image2.jpeg", cv2.IMREAD_GRAYSCALE)

low = 0  
high = 50  

uniform_noise = np.random.uniform(low, high, size=img.shape).astype('uint8')
    
noisy_image = cv2.addWeighted(img, 0.5, uniform_noise, 0.5, 0)

noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')

cv2.imwrite('image_Uniform_noise.jpeg', noisy_image)

cv2.imshow("Original Image", img)
cv2.imshow("Noisy Image with Uniform Noise", noisy_image)



#Press q or esc to exit;
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break


cv2.destroyAllWindows()
