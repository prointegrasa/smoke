

import cv2


def standardize_image_size(img, STANDARDIZED_IMAGE_WIDTH, STANDARDIZED_IMAGE_HEIGHT):

    border_v = 0
    border_h = 0
    if (STANDARDIZED_IMAGE_WIDTH / STANDARDIZED_IMAGE_HEIGHT) >= (img.shape[0] / img.shape[1]):
        border_v = int((((STANDARDIZED_IMAGE_WIDTH / STANDARDIZED_IMAGE_HEIGHT) * img.shape[1]) - img.shape[0]) / 2)
    else:
        border_h = int((((STANDARDIZED_IMAGE_HEIGHT / STANDARDIZED_IMAGE_WIDTH) * img.shape[0]) - img.shape[1]) / 2)
    img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
    img = cv2.resize(img, (STANDARDIZED_IMAGE_HEIGHT, STANDARDIZED_IMAGE_WIDTH))

    return img


def preprocess_image(img):
    # apply some preprocessing - gaussian blur

    # result = cv2.blur(img, (5, 5))

    return img

