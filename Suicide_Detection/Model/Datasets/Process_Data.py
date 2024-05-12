from pathlib import Path

import cv2
import os
import shutil


def process_image(image_path: str, label_path: str, save_path: str, mode: str):
    '''
    Process the image and label to the save path with the same name
    '''

    global counter

    image, label = os.listdir(image_path), os.listdir(label_path)
    image.sort(); label.sort()

    # Create the directories if not found
    # The following can be combined into one, but for clarity, it is separated
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(f'{save_path}/{mode}'):
        os.mkdir(f'{save_path}/{mode}')

    if not os.path.exists(f'{save_path}/{mode}/images'):
        os.mkdir(f'{save_path}/{mode}/images')

    if not os.path.exists(f'{save_path}/{mode}/labels'):
        os.mkdir(f'{save_path}/{mode}/labels')

    for i in range(len(image)):
        image_name = image[i]
        label_name = label[i]

        if not os.path.exists(f'{image_path}/{counter}.jpg'):
            os.rename(image_path + '/' + image_name, f'{image_path}/{counter}.jpg')
        if not os.path.exists(f'{label_path}/{counter}.txt'):
            os.rename(label_path + '/' + label_name, f'{label_path}/{counter}.txt')

        img = cv2.imread(f'{image_path}/{counter}.jpg')
        img = cv2.resize(img, (480, 480))
        cv2.imwrite(f'{save_path}/{mode}/images/{counter}.jpg', img)

        shutil.copy(f'{label_path}/{counter}.txt', f'{save_path}/{mode}/labels/{counter}.txt')

        counter += 1


if __name__ == '__main__':
    counter = 0 # Image name

    # Paths
    file_path = Path(__file__).resolve().parent

    d1_path = str(file_path / 'Raw/Extracted/D1')
    d2_path = str(file_path / 'Raw/Extracted/D2')
    d3_path = str(file_path / 'Raw/Extracted/D3')

    save_path = str(file_path / 'Processed')

    # Processed and resized images
    process_image(d1_path + '/train/images', d1_path + '/train/labels', save_path, 'train')
    process_image(d1_path + '/valid/images', d1_path + '/valid/labels', save_path, 'val')
    process_image(d1_path + '/test/images', d1_path + '/test/labels', save_path, 'test')

    process_image(d2_path + '/train/images', d2_path + '/train/labels', save_path, 'train')

    process_image(d3_path + '/train/images', d3_path + '/train/labels', save_path, 'train')
    process_image(d3_path + '/valid/images', d3_path + '/valid/labels', save_path, 'val')
    process_image(d3_path + '/test/images', d3_path + '/test/labels', save_path, 'test')