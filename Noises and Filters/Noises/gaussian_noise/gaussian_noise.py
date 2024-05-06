import cv2
import numpy as np



img = cv2.imread("../../image2.jpeg", cv2.IMREAD_GRAYSCALE)


mean = 0
std_dev = 20
noise = np.random.normal(mean, std_dev, img.shape).astype('uint8')


noisy_image = cv2.add(img, noise)


cv2.imwrite('image_gaussian_noise.jpeg', noisy_image)


cv2.imshow("Original Image", img)
cv2.imshow("Noisy Image", noisy_image)



#Press q or esc to exit;
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break


cv2.destroyAllWindows()

