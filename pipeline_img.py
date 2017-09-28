import cv2
import glob
import pickle
from modules.lesson_functions import *
from modules.find_cars import *
from modules.spatial_configs import *
from scipy.ndimage.measurements import label
from tqdm import *

test_imgs = glob.glob('test_images/*.jpg')

# Getting back the calibration points:
with open('training_data/trained_svm.pickle','rb') as f:
    svc,X_scaler = pickle.load(f)

for i in tqdm(range(len(test_imgs))):
    fname = test_imgs[i]
    image = cv2.imread(fname)[:,:,::-1]
    heat_img = np.copy(image)
    draw_image = np.copy(image)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    color_list = [(255, 0, 0),(0, 155, 0),(0, 255, 0),(0, 0, 255)]

    # create sliding windows
    window_list_far, slide_far   = window_list(image, x = [0,1280], y = [400,500], w = [32, 32], c = color_list[0])
    window_list_med1, slide_med1 = window_list(image, x = [0,1280], y = [390,570], w = [64, 64], c = color_list[1])
    window_list_med2, slide_med2 = window_list(image, x = [0,1280], y = [380,640], w = [96, 96], c = color_list[2])
    window_list_near, slide_near = window_list(image, x = [0,1280], y = [370,720], w = [128, 128], c = color_list[3])

    cv2.imwrite('output_images/image_slide_windows/img{}_far.png'.format(i), slide_far)
    cv2.imwrite('output_images/image_slide_windows/img{}_med1.png'.format(i), slide_med1)
    cv2.imwrite('output_images/image_slide_windows/img{}_med2.png'.format(i), slide_med2)
    cv2.imwrite('output_images/image_slide_windows/img{}_near.png'.format(i), slide_near)

    window_lists = [window_list_far,window_list_med1,window_list_med2,window_list_near]

    for w in range(len(window_lists)):
        hot_windows = search_windows(image, window_lists[w], svc, X_scaler, color_space=colorspace,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block, hog_channel=hog_channel,
                                spatial_size=spatial_size, hist_bins=hist_bins)

        draw_image = draw_boxes(draw_image, hot_windows, color=color_list[3], thick=3)
        heat = add_heat(heat,hot_windows)
        heat = apply_threshold(heat,1)
        heatmap  = np.clip(heat, 0, 255)
        labels   = label(heatmap)
        heat_img = draw_labeled_bboxes(heat_img, labels)

    cv2.imwrite('output_images/found_cars/img{}.png'.format(i), draw_image)
    cv2.imwrite('output_images/heatmap_cars/img{}.png'.format(i), heat_img)
