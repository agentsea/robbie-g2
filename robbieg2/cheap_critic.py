import cv2
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

def assess_action_result(starting_image: Image.Image, updated_image: Image.Image) -> (float, bool):
    """Cheap critic returns True if the chain of actions can be continued and False otherwise.
    In the current version, we continue if the SSIM is above a threshold (i.e. the images are visually similar).
    """
    threshold = 0.9
    ssim = compare_images(starting_image, updated_image)
    if ssim > threshold:
        return ssim, True
    else:
        return ssim, False

def _pil_to_cv2(pil_image):
    # Ensure the image is in RGB mode
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    # Convert the PIL Image to a NumPy array
    np_image = np.array(pil_image)
    # Convert RGB to BGR format (OpenCV uses BGR)
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return cv2_image

def compare_images(image1, image2):
    image1 = _pil_to_cv2(image1)
    image2 = _pil_to_cv2(image2)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    similarity_index, diff = ssim(gray1, gray2, full=True)

    print(f"SSIM: {similarity_index}")
    return similarity_index
