import dlib
from os import listdir, makedirs
from os.path import isfile, join, basename, exists, isdir
from tqdm import tqdm
from argparse import ArgumentParser


def main(config):
    detector_ori = dlib.get_frontal_face_detector()

    # open the image file
    try:
        img = dlib.load_rgb_image(config.input_path)
    except Exception as e:
        print("While processing, " + str(e) + '\n')
        exit()

    # If the resolution is less than 128x128 then skip
    img_height = img.shape[0]
    img_width = img.shape[1]
    if img_height < 128 or img_width < 128:
        print("While processing, image size too small" + '\n')
        exit()

    # find one face that best matches and finalize the image cropping size
    max_object = None

    dets, score, idx = detector_ori.run(img, 1, -1)
    max_confi = 0.6

    if len(dets) == 0:
        print("While processing, face not detected" + '\n')
        exit()

    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
        if max_confi < score[i]:
            max_confi = score[i]
            max_object = d

    d = max_object

    if d == None:
        print("While processing, face not detected" + '\n')
        exit()
    d_width = int((d.right() - d.left() + 1) // 2)
    d_height = int((d.bottom() - d.top() + 1) // 2)

    crop_top = d.top() - d_height
    crop_bottom = d.bottom() + d_height
    crop_left = d.left() - d_width
    crop_right = d.right() + d_width

    img_out_lenght = min(crop_top, crop_left, img_height - crop_bottom, img_width - crop_right)

    if img_out_lenght < -d_width / 2:
        print("While processing, face image over index" + '\n')
        exit()

    if img_out_lenght < 0:
        crop_top = crop_top - img_out_lenght
        crop_bottom = crop_bottom + img_out_lenght
        crop_left = crop_left - img_out_lenght
        crop_right = crop_right + img_out_lenght

    # Make the cropped and resized image from the original one
    img = img[crop_top:crop_bottom, crop_left:crop_right]
    if img.shape[0] != img.shape[1]:
        final_size = min(img.shape[0], img.shape[1])
        img = img[:final_size, :final_size]
    img = dlib.resize_image(img, config.image_size / img.shape[0])

    dlib.save_image(img, config.save_path)


if __name__ == '__main__':
    parser = ArgumentParser()

    # argument setting
    parser.add_argument('--input_path', type=str, default="trainset/fake/frame1.jpg")
    parser.add_argument('--save_path', type=str, default= "cropped_trainset/fake/frame1.png")
    parser.add_argument('--model_path', type=str, default='mmod_human_face_detector.dat')
    parser.add_argument('--image_size', type=int, default=128)

    config = parser.parse_args()

    if config.input_path == None or config.save_path == None:
        print("insert parameter")
        exit()

    main(config)