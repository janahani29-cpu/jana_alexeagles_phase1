import cv2 as cv
import numpy as np

def preprocess(image_path):
    img = cv.imread(image_path)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img, (11,11), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, 125, 175)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    return img, gray, blur, canny, thresh


def compare_with_ideal(sample_path, ideal_thresh, ideal_shape):
    sample_img, gray_sample, blur_sample, canny_sample, thresh_sample = preprocess(sample_path)

    # XOR difference
    diff = cv.bitwise_xor(ideal_thresh, thresh_sample)

    # Mask out center circle
    h, w = diff.shape
    white_img = np.ones((h, w), dtype="uint8") * 255
    center = (w // 2, h // 2)
    radius = w // 7
    cv.circle(white_img, center, radius, 0, -1)
    diff_masked = cv.bitwise_and(diff, white_img)

    # Clean up with morphology
    kernel = np.ones((3,3), np.uint8)
    diff_clean = cv.morphologyEx(diff_masked, cv.MORPH_OPEN, kernel)
    diff_clean = cv.morphologyEx(diff_clean, cv.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv.findContours(diff_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    blank = np.zeros(sample_img.shape, dtype='uint8')

    for c in contours: #this loop is made with help of copilot
        area = cv.contourArea(c)
        perimeter = cv.arcLength(c, True)

        # Skip tiny noise
        if area < 50:
            continue

        if area >400:  # large missing chunk
            defect_type = "Broken"
            color = (0,0,255)   # Red
        else:           # small, smoother edge
            defect_type = "Worn"
            color = (0,255,0)   # Green

        cv.drawContours(blank, [c], -1, color, 2)
        x,y,wc,hc = cv.boundingRect(c)
        cv.putText(blank, defect_type, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1)

    # Show results
    cv.imshow("XOR Differences", diff)
    cv.imshow("Masked Differences", diff_masked)
    cv.imshow("Defect Classification", blank)

    print(f"{sample_path} → Defects classified: {len(contours)}")
    cv.waitKey(0)


ideal_img, grayideal, blurideal, cannyideal, threshideal = preprocess("ideal.jpg")
samples = ["sample2.jpg", "sample3.jpg", "sample4.jpg", "sample5.jpg", "sample6.jpg"]

for s in samples:
    compare_with_ideal(s, threshideal, ideal_img.shape)

cv.destroyAllWindows()
