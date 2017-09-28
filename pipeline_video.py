from moviepy.editor import VideoFileClip
from modules.lesson_functions import *
from modules.find_cars import *
from modules.spatial_configs import *
from scipy.ndimage.measurements import label
from collections import deque
import pickle
global i
i = -1
def apply_pipeline(image, svc, X_scaler, g_heat):
    global i
    i+=1
    # heat_img = np.copy(image)
    draw_image = np.copy(image)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    color_list = [(255, 0, 0),(0, 155, 0),(0, 255, 0),(0, 0, 155),(0, 0, 255)]

    # create sliding windows
    window_list_far, slide_far   = window_list(image, x = [100,1280], y = [400,528], w = [64, 64], c = color_list[0])
    window_list_med1, slide_med1 = window_list(image, x = [100,1280], y = [375,564], w = [128, 128], c = color_list[1])
    # window_list_med2, slide_med2 = window_list(slide_med1, x = [100,1280], y = [350,700], w = [155, 155], c = color_list[2])
    window_list_near, slide_near = window_list(image, x = [100,1280], y = [350,700], w = [192, 192], c = color_list[3])
    # window_list_near2, slide_near2 = window_list(image, x = [100,1280], y = [360,720], w = [120, 120], c = color_list[4])

    # window_lists = [window_list_far,window_list_med1,window_list_med2,window_list_near,window_list_near2]
    window_lists = [window_list_far,window_list_med1,window_list_near]

    for w in range(len(window_lists)):
        hot_windows = search_windows(image, window_lists[w], svc, X_scaler, color_space=colorspace,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block, hog_channel=hog_channel,
                                spatial_size=spatial_size, hist_bins=hist_bins)

        # draw_image = draw_boxes(draw_image, hot_windows, color=color_list[3], thick=2)
        heat = add_heat(heat, hot_windows)

    g_heat.append(heat) #save for future frames

    for old_heat in g_heat:
        heat = heat + old_heat #add heat from old frames

    heatmap  = np.clip(heat, 0, 255)

    if i > 8 :
        heatmap  = heatmap/10
    heatmap  = apply_threshold(heatmap,1)
    labels   = label(heatmap)
    heat_img = draw_labeled_bboxes(image, labels)

    # mpimg.imsave('output_images/heatmap_cars/img{}.png'.format(i),heatmap)

    # return draw_image
    return heat_img

# Getting back the calibration points:
with open('training_data/trained_svm.pickle','rb') as f:
    svc,X_scaler = pickle.load(f)

g_heat = deque(maxlen=10)
pipeLambda = lambda img: apply_pipeline(img, svc, X_scaler, g_heat)

# clip = VideoFileClip("project_video.mp4").subclip(10,20)
# clip = VideoFileClip("test_video.mp4")
clip = VideoFileClip("project_video.mp4")
processed_clip = clip.fl_image(pipeLambda)
processed_clip.write_videofile("project_processed_video.mp4", audio=False)
