import cv2 as cv
import numpy as np

try:
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass

from functools import reduce
import subprocess
import pickle

WEBCAM_CAPTURE_CHAR = 'c'

N_CHARS = 9

PUZZLE_SIZE_DIMEN = 450
PUZZLE_SIZE = (PUZZLE_SIZE_DIMEN, PUZZLE_SIZE_DIMEN)
EXTRACTED_DIGIT_DIMEN = 30
EXTRACTED_DIGIT_SIZE = (EXTRACTED_DIGIT_DIMEN, EXTRACTED_DIGIT_DIMEN)

BOX_DIMEN = PUZZLE_SIZE_DIMEN / N_CHARS

PP_BLUR_KERNEL_SIZE = (5, 5)

DX_DILATE_KERNEL_SIZE = (3, 3)
DX_ERODE_KERNEL_SIZE = (5, 5)

MODEL_LOCATION = "../digit-recog/knn.model"
SOLVER_LOCATION = "../sudoku-solver/sudoku-solver"

TEST_IMG_DIR = "../test-images"

COL_BLUE = (255, 0, 0)
COL_GREEN = (0, 255, 0)
COL_WHITE = (255, 255, 255)

BOUNDARY_COLOR = COL_BLUE
DX_BORDER_COLOR = COL_WHITE
DX_BOX_COLOR = COL_GREEN

def get_sudoku_image(path):
    img = cv.imread(path)
    
    if img is not None:
        return img
    else:
        
        WIN_NAME = 'live'

        cap = cv.VideoCapture(0)
        cv.namedWindow(WIN_NAME)

        while True:
            ret, frame = cap.read()

            cv.imshow(WIN_NAME, frame)

            if cv.waitKey(1) & 0xFF == ord(WEBCAM_CAPTURE_CHAR):
                img = frame
                break

        cap.release()
        cv.destroyWindow(WIN_NAME)

        return img

def pre_process(img):
    if len(img.shape) == 3:
        proc = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    else:
        proc = img

    proc = cv.GaussianBlur(proc, PP_BLUR_KERNEL_SIZE, 0)
    proc = cv.adaptiveThreshold(proc, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    
    return proc

def get_largest_contour(contours):
    return max(contours, key=lambda cnt: cv.contourArea(cnt))

def get_contour_corners(contour):
    sum_xy = [pt[0][0] + pt[0][1] for pt in contour]
    diff_xy = [pt[0][0] - pt[0][1] for pt in contour]
    
    indices = [
        np.argmin(sum_xy),
        np.argmax(diff_xy),
        np.argmax(sum_xy),
        np.argmin(diff_xy)
    ]
    
    return [contour[i][0] for i in indices]

def get_corners(img):
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

    puzzle_bound = get_largest_contour(contours)

    return get_contour_corners(puzzle_bound)

def extract_rect(img, corners, final_size=PUZZLE_SIZE):
    dist = lambda p1, p2: np.linalg.norm(np.array(p1) - np.array(p2))

    tl, tr, br, bl = corners

    side = max([dist(tl, tr), dist(tr, bl), dist(br, bl), dist(bl, tl)])

    src = np.array([tl, tr, br, bl], 'float32')
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], 'float32')

    m = cv.getPerspectiveTransform(src, dst)
    puzzle = cv.warpPerspective(img, m, (int(side), int(side)))
    
    puzzle = cv.resize(puzzle, final_size)

    return puzzle

def mark_boundary(img, corners):
    """
    This is purely for visualisation purposes.
    
    This draws a box on the image with the specified corners
    """

    tl, tr, br, bl = [tuple(x) for x in corners]
    
    cv.line(img, tl, tr, BOUNDARY_COLOR, 3)
    cv.line(img, tr, br, BOUNDARY_COLOR, 3)
    cv.line(img, br, bl, BOUNDARY_COLOR, 3)
    cv.line(img, bl, tl, BOUNDARY_COLOR, 3)
    
    return img

def is_mostly_empty(img, thresh=0.1):
    total = img.shape[0] * img.shape[1]
    white = np.count_nonzero(img)
    
    return white / total < thresh

def approx_center(center):
    """
    Approxiamates the box's center to help
        in determining the order in which
        they originally came in the puzzle
    """
    
    half_box = BOX_DIMEN / 2
    
    valid_centers = np.arange(half_box, PUZZLE_SIZE_DIMEN - half_box + 1, BOX_DIMEN)

    cx, cy = center
    
    x = valid_centers[np.argmin(np.abs(valid_centers - cx))]
    y = valid_centers[np.argmin(np.abs(valid_centers - cy))]
    
    return (x, y)
        

def get_digits(img):        
    img = cv.dilate(img, np.ones(DX_DILATE_KERNEL_SIZE, np.uint8))
    img = cv.erode(img, np.ones(DX_ERODE_KERNEL_SIZE, np.uint8))
    img = cv.copyMakeBorder(img, 5, 5, 5, 5, cv.BORDER_CONSTANT, None, DX_BORDER_COLOR)
    
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)        
    cp_img_color = img_color.copy()
    gridless = img_color.copy()
    
    edges = cv.Canny(img, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 150)
    
    for line in lines:
        rho, theta = line[0]
        
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 3) 
        cv.line(gridless, (x1, y1), (x2, y2), (0, 0, 0), 5)
    
    boxes = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    boxes = cv.erode(boxes, np.ones(DX_ERODE_KERNEL_SIZE, np.uint8))
    boxes = cv.dilate(boxes, np.ones(DX_DILATE_KERNEL_SIZE, np.uint8))
    
    contours = cv.findContours(boxes, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]
    
    contours = sorted(contours, reverse=True, key=cv.contourArea)[1:82]
    
    digits = []
    cv.drawContours(cp_img_color, contours, -1, DX_BOX_COLOR, 3)
    
    for cnt in contours:
        corners = get_contour_corners(cnt)        
        digit = extract_rect(gridless, corners, EXTRACTED_DIGIT_SIZE)
        digit = cv.dilate(digit, np.ones(DX_DILATE_KERNEL_SIZE, np.uint8))

        digit = cv.cvtColor(digit, cv.COLOR_BGR2GRAY)
        _, digit = cv.threshold(digit, 75, 255, cv.THRESH_BINARY)
        
        center = approx_center(list(np.array(reduce(lambda x1, x2: [x1[0] + x2[0], x1[1] + x2[1]], corners)) / 4))

        digits.append({ "img": digit, "center": center })
        
    return sorted(digits, key=lambda x: [x['center'][1], x['center'][0]]), cp_img_color

def wait_for_key(wait_char):
    while True:
        if cv.waitKey(1) & 0xFF == ord(wait_char):
            break

def display_img(img, name='Image', wait_key=None):
    cv.imshow(name, img)
    
    if wait_key is None:
        cv.waitKey(0)
    else:
        wait_for_key(wait_key)

def is_running():
    return __name__ == '__main__' and '__file__' not in globals()

def  get_model():
    model_pickle = open(MODEL_LOCATION, 'rb')
    model = pickle.load(model_pickle)
    model_pickle.close()
    
    return model

def classify_img(model, img):
    return str(model.predict([img.flatten()])[0])

def solve(puzzle):
    solver = subprocess.Popen([SOLVER_LOCATION], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    solver.stdin.write((puzzle + "\n").encode())

    out = solver.communicate()[0]
    
    ret_val = solver.returncode
    
    solver.stdin.close()
    
    solver.terminate()
    
    return out.decode("utf-8"), ret_val == 0

def solve_visudoku(img_path=None, debug=False):
    
    img_stages = []
    
    img = get_sudoku_image(img_path)
    
    img_stages.append({ "label": "Input", "img": img.copy() })
    
    if debug:
        pass
        pass

    copy = img

    img = pre_process(img)
    
    img_stages.append({ "label": "Post Preprocessing", "img": img.copy() })
    
    if debug:
        pass
        pass

    corners = get_corners(img)
    marked = mark_boundary(copy, corners)
    
    img_stages.append({ "label": "Boundaries marked", "img": marked.copy() })

    if debug:
        pass
        pass
    
    puzzle = extract_rect(img, corners)
    
    img_stages.append({ "label": "Extracted puzzle", "img": puzzle.copy() })
    
    if debug:
        pass
        pass
    
    digits, boxed_digits = get_digits(puzzle)
    
    img_stages.append({ "label": "Boxed digits", "img": boxed_digits.copy() })
    
    if debug:
        pass
        pass
    
    model = get_model()
    nums = []
    
    for dig in digits:
        num = classify_img(model, dig['img'])
        
        if debug:
            print(num)
            pass
            pass
        
        nums.append(classify_img(model, dig['img']))
        
    solution, success = solve(' '.join(nums))
    
    return solution, success, img_stages, nums

    
if is_running():
    for i in range(6):
        img_path = f"{TEST_IMG_DIR}/test{i}.jpg"
        
        img = cv.imread(img_path)
        pass
        pass
        
        solution, success, img_stages, nums = solve_visudoku(img_path)
        
        if success:
            print(solution)
        else:
            print("Oops, something went wrong!")
            print("It will soon be possible to correct this mistake!")

