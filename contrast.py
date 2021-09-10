import cv2 as cv
import time
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2 as cv
#cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap = cv.VideoCapture(0)
a = 0
s = 128
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
blist = [0, -127, 127,   0,  0, 64] # list of brightness values
clist = [0,    0,   0, -64, 64, 64]
out = np.zeros((s*2, s*3, 3), dtype = np.uint8)
while True:
        a = a+1
        check, frame = cap.read()
        print(check)
        print(frame)
        b,g,r,a = 0,255,0,0
        font = ImageFont.truetype("arial.ttf",32)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((50, 80), "ШОКОЛАД", font = font, fill = (b,g,r,a))
        frame = np.array(img_pil)
        
        #sharpened_image = unsharp_mask(frame)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        kernel2 = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        kernel3 = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, 0]], np.float32)
        kernel3 = 1/2 * kernel3

        #frame = cv.filter2D(frame, -1, kernel)
        #frame = cv.convertScaleAbs(frame, alpha=1.5, beta=0)
        frame = apply_brightness_contrast (frame, 0, 64)
        #cv.addWeighted(frame, 1.5, frame, -0.5, 0, frame);
        cv.imshow('CApt', frame)

        key = cv.waitKey(1)
        if key == ord('q'):
                break
print(a)
cap.release()
cv.destroyAllWindows
