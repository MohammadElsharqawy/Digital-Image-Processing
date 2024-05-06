import cv2
import numpy as np

def arithmetic_mean_filter(img, size):
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return cv2.filter2D(img, -1, kernel)

def harmonic_mean_filter(img, size):
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return cv2.filter2D(1 / (cv2.filter2D(1 / img.astype(np.float32), -1, kernel)), -1, kernel)

def geometric_mean_filter(img, size):
    return cv2.pow(cv2.filter2D(cv2.pow(img.astype(np.float32), 1/size), -1, np.ones((size, size), dtype=np.float32) / (size * size)), size)

def contraharmonic_mean_filter(img, size, Q):
    numerator = cv2.filter2D(cv2.pow(img.astype(np.float32), Q+1), -1, np.ones((size, size), dtype=np.float32) / (size * size))
    denominator = cv2.filter2D(cv2.pow(img.astype(np.float32), Q), -1, np.ones((size, size), dtype=np.float32) / (size * size))
    return numerator / (denominator + np.finfo(float).eps)

def min_filter(img, size):
    return cv2.erode(img, np.ones((size, size), dtype=np.uint8))

def max_filter(img, size):
    return cv2.dilate(img, np.ones((size, size), dtype=np.uint8))

def midpoint_filter(img, size):
    min_img = cv2.erode(img, np.ones((size, size), dtype=np.uint8))
    max_img = cv2.dilate(img, np.ones((size, size), dtype=np.uint8))
    return (min_img + max_img) // 2

def alpha_trimmed_mean_filter(img, size, d):
    trimmed_img = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_REPLICATE)
    for i in range(d, img.shape[0] + d):
        for j in range(d, img.shape[1] + d):
            neighbors = trimmed_img[i-d:i+d+1, j-d:j+d+1]
            sorted_neighbors = np.sort(neighbors.flatten())
            trimmed_neighbors = sorted_neighbors[d:-d]
            trimmed_img[i, j] = np.mean(trimmed_neighbors)
    return trimmed_img[d:-d, d:-d]




if __name__ == '__main__':
    # Load the noisy images
    noisy_image_Erlang = cv2.imread("./Noises/Erlang(Gamma)_noise/image_Erlang(Gamma)_noise.jpeg", cv2.IMREAD_GRAYSCALE)
    noisy_image_Exponential = cv2.imread("./Noises/Exponential_noise/image_exponential_noise.jpeg", cv2.IMREAD_GRAYSCALE)
    noisy_image_gaussian = cv2.imread("./Noises/gaussian_noise/image_gaussian_noise.jpeg", cv2.IMREAD_GRAYSCALE)
    noisy_image_rayleight = cv2.imread("./Noises/rayleight_noise/image_rayleigh_noise.jpeg", cv2.IMREAD_GRAYSCALE)
    noisy_image_SaltAndPepper = cv2.imread("./Noises/SaltAndPepper_noise/image_SaltAndPepper_noise.jpeg", cv2.IMREAD_GRAYSCALE)
    noisy_image_Uniform = cv2.imread("./Noises/Uniform_noise/image_Uniform_noise.jpeg", cv2.IMREAD_GRAYSCALE)

    Noisy_images= [noisy_image_Erlang,noisy_image_Exponential, noisy_image_gaussian, noisy_image_rayleight, noisy_image_SaltAndPepper, noisy_image_Uniform]
    Noise_name = ["Erlang Noise", "Exponential Noise", "Gaussian Noise", "rayleight Noise","SaltAndPepper Noise","Uniform Noise"  ]
    
    
    # Apply filters

    ## HINTS / CAUSTIOUS ------
    #For easy Testing UNHASH only one filter at a time 
    # test the image generated, preferably with vscoede or any other media player but not with CV2 for better images.


    # #apply arithmetic mean filter to all images.
    # for i in range(len(Noisy_images)):
    #     arithmetic_mean_filtered = arithmetic_mean_filter(Noisy_images[i], size=3)
    #     cv2.imwrite(f"Arithmetic Mean Filter on Image with {Noise_name[i]}.jpeg", Noisy_images[i])
    #     cv2.imshow(f"Arithmetic Mean Filter on Image with {Noise_name[i]}", arithmetic_mean_filtered)


    # #apply Harmonic mean filter to all images.
    # for i in range(len(Noisy_images)):
    #     harmonic_mean_filtered = harmonic_mean_filter(Noisy_images[i], size=3)
    #     cv2.imwrite(f"Harmonic Mean  Filter on Image with {Noise_name[i]}.jpeg", Noisy_images[i])
    #     cv2.imshow(f"Harmonic Mean Filter on Image with {Noise_name[i]}", harmonic_mean_filtered)

    # #apply Geometric mean filter to all images.
    # for i in range(len(Noisy_images)):
    #     geometric_mean_filtered = geometric_mean_filter(Noisy_images[i], size=3)
    #     cv2.imwrite(f"Geometric Mean  Filter on Image with {Noise_name[i]}.jpeg", Noisy_images[i])
    #     cv2.imshow(f"Geometric Mean Filter on Image with {Noise_name[i]}", geometric_mean_filtered)

    # #apply contraharmonic mean filter to all images with Q =1.
    # for i in range(len(Noisy_images)):
    #     contraharmonic_mean_filtered = contraharmonic_mean_filter(Noisy_images[i], size=3, Q=1)
    #     cv2.imwrite(f"contraharmonic Mean Q = 1 Filter on Image with {Noise_name[i]}.jpeg", Noisy_images[i])
    #     cv2.imshow(f"contraharmonic Mean Filter on Image with {Noise_name[i]}", contraharmonic_mean_filtered)

    # #apply contraharmonic mean filter to all images with Q = -1.
    # for i in range(len(Noisy_images)):
    #     contraharmonic_mean_filtered = contraharmonic_mean_filter(Noisy_images[i], size=3, Q=-1)
    #     cv2.imwrite(f"contraharmonic Mean Q = -1 Filter on Image with {Noise_name[i]}.jpeg", Noisy_images[i])
    #     cv2.imshow(f"contraharmonic Mean Filter on Image with {Noise_name[i]}", contraharmonic_mean_filtered)

    # #apply Min filter to all images.
    # for i in range(len(Noisy_images)):
    #     min_filtered = min_filter(Noisy_images[i], size=3)
    #     cv2.imwrite(f"Min Filter on Image with {Noise_name[i]}.jpeg", Noisy_images[i])
    #     cv2.imshow(f"Min Filter on Image with {Noise_name[i]}", min_filtered)

    # #apply Max filter to all images.
    # for i in range(len(Noisy_images)):
    #     max_filtered = max_filter(Noisy_images[i], size=3)
    #     cv2.imwrite(f"Max Filter on Image with {Noise_name[i]}.jpeg", Noisy_images[i])
    #     cv2.imshow(f"Max Filter on Image with {Noise_name[i]}", max_filtered)

    # #apply midpoint filter to all images.
    # for i in range(len(Noisy_images)):
    #     midpoint_filtered = midpoint_filter(Noisy_images[i], size=3)
    #     cv2.imwrite(f"midpoint Filter on Image with {Noise_name[i]}.jpeg", Noisy_images[i])
    #     cv2.imshow(f"midpoint Filter on Image with {Noise_name[i]}", midpoint_filtered)
    
    #  #apply alpha_trimmed filter to all images.
    # for i in range(len(Noisy_images)):
    #     alpha_trimmed_filtered = alpha_trimmed_mean_filter(Noisy_images[i], size=3, d=2)
    #     cv2.imwrite(f"alpha_trimmed Filter on Image with {Noise_name[i]}.jpeg", Noisy_images[i])
    #     cv2.imshow(f"Alpha Trimmed Filter on Image with {Noise_name[i]}", alpha_trimmed_filtered)
    
    
    #Press q or esc to exit;
    while True:
        # Wait for a key event for a specified time (1 millisecond)
        key = cv2.waitKey(1) & 0xFF
        
        # Check if the X button or the 'q' key is pressed
        if key == ord("q") or key == 27:  # 27 is the ASCII code for the Esc key
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()


    