import cv2
import numpy as np


img = cv2.imread("../../image2.jpeg", cv2.IMREAD_GRAYSCALE)


shape = 2  
scale = 50  
alpha = 0.5 

erlang_noise = np.random.gamma(shape, scale, size=img.shape).astype('uint8')
    
noisy_image = cv2.addWeighted(img, 1 - alpha, erlang_noise, alpha, 0)
    
noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')

cv2.imwrite('image_Erlang(Gamma)_noise.jpeg', noisy_image)


cv2.imshow("Original Image", img)
cv2.imshow("Noisy Image with Erlang Noise", noisy_image)



#Press q or esc to exit;
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break


cv2.destroyAllWindows()