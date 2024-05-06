import numpy as np
import cv2


img = cv2.imread("../../image2.jpeg", cv2.IMREAD_GRAYSCALE).astype(np.float64)

noise_std = 0.2
noise = np.random.rayleigh(noise_std, img.shape)


noisy_image = cv2.addWeighted(img, .5, noise, 0.5, 0.0).astype(np.uint8)


cv2.imwrite('image_rayleigh_noise.jpeg', noisy_image)


cv2.imshow('Image', img)
cv2.imshow("Noise", noise)
cv2.imshow("Noisy Image", noisy_image)


#Press q or esc to exit;
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:  
        break


cv2.destroyAllWindows()

