# PlanBrainer

PlanBrainer is a project designed to understand and implement Generative Adversarial Networks (GANs) by creating optimized interior floor plans. The objective is to generate interior floor plan layouts based on user-defined window and door positions, utilizing Conditional GANs (CGANs).

While comprehensive documentation is underway, I'm currently rewriting the entire codebase in Object-Oriented Programming (OOP) paradigm. Given the niche nature of interior floor plan data, I crafted a custom dataset from scratch, both to address the lack of suitable existing datasets and as a hands-on learning experience which you can find below.

## Pre-processing Images

### Type of images issue:

First of all the website from which the images were scrapped contained some random floor plans which weren’t suitable for the style of floor plans I was going for.

![new](https://github.com/pruthvik-sheth/PlanBrainer/assets/80819203/78248783-e41b-4281-9d9a-20495856d71e)


To address this issue I implemented a filtering mechanism while downloading the images
To filter out such images (undesired images) I compared each and every image (after it is downloaded) to an random desired image.

```python
def check_similarity(incoming_img):

    img1 = sample_image
    img2 = incoming_img

    # img1 = cv.resize(img1, [1000, 1000])
    img2 = cv.resize(img2, [1000, 1000])

    mse_r = np.mean((img1[:,:,0] - img2[:,:,0])**2)
    mse_g = np.mean((img1[:,:,1] - img2[:,:,1])**2)
    mse_b = np.mean((img1[:,:,2] - img2[:,:,2])**2)

    # Calculate the total MSE across all channels
    return ((mse_r + mse_g + mse_b) / 3)
```

In the above code I was checking the Mean Square Error on all the channels (Red, Green, Blue) of the new incoming downloaded image with the desired image. This function will return the MSE score of the new image and based on that threshold I are filtering out images.

```python
score = check_similarity(image)

        if score > 60:
            print('\\nTrash image found')
            file_path = os.path.join(DOWNLOAD_PATH_TRASH, file_name)
        else:
            print('\\nGood image found')
            file_path = os.path.join(DOWNLOAD_PATH_GOOD, file_name)
```

Now the entire image downloader script which takes a links.csv file and downloads and sorts each and every image from the url present in the CSV file.

```python
import pathlib
import pandas as pd
import requests as req
from PIL import Image
import io
import os
import time
import numpy as np
import cv2 as cv

PAGE_START = 21
PAGE_END = 31
LAST_IMAGE_DOWNLOADED_INDEX = 364
CSV_PATH = pathlib.Path(f'csv_files/links_records_{PAGE_START}_to_{PAGE_END}.csv')
SAMPLE_IMAGE_PATH = pathlib.Path('bechmark_image/')
DOWNLOAD_PATH_GOOD = pathlib.Path(f'downloaded_images/good_ones_{PAGE_START}_to_{PAGE_END}/')
DOWNLOAD_PATH_TRASH = pathlib.Path(f'downloaded_images/trash_ones_{PAGE_START}_to_{PAGE_END}/')

df = pd.read_csv(CSV_PATH)
links = df['Image_link']

# Image to be compared with (Desired Image)
sample_image = cv.imread(os.path.join(SAMPLE_IMAGE_PATH,'sample.png'))

def check_similarity(incoming_img):

    img1 = sample_image
    img2 = incoming_img

    # img1 = cv.resize(img1, [1000, 1000])
    img2 = cv.resize(img2, [1000, 1000])

    mse_r = np.mean((img1[:,:,0] - img2[:,:,0])**2)
    mse_g = np.mean((img1[:,:,1] - img2[:,:,1])**2)
    mse_b = np.mean((img1[:,:,2] - img2[:,:,2])**2)

    # Calculate the total MSE across all channels
    return ((mse_r + mse_g + mse_b) / 3)

def download_image(url, file_name):
    try:
        image_content = req.get(url).content
        image_file = io.BytesIO(image_content)
        pil_image = Image.open(image_file)

        image = np.array(pil_image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        score = check_similarity(image)

        if score > 60:
            print('\\nTrash image found')
            file_path = os.path.join(DOWNLOAD_PATH_TRASH, file_name)
        else:
            print('\\nGood image found')
            file_path = os.path.join(DOWNLOAD_PATH_GOOD, file_name)

        with open(file_path, "wb") as f:
            pil_image.save(f, "PNG")

        print("Downloaded: ", file_name)
    except Exception as e:
        print("FAILED -", e)

# Downloading and Filtering all images
for index, link in enumerate(links, LAST_IMAGE_DOWNLOADED_INDEX):
    download_image(link, str(index + 1) + '.png')
    time.sleep(1.5)

    if (index + 1) % 10 == 0:
        time.sleep(5)
```

### Size of Images issue:

Now, the images are filtered the next problem was to make the sizes of all the downloaded images even. As the downloaded images were of different sizes but mostly 1000X750 they needed to be resized.

But, if the images are resized directly then I might loose the aspect ratio of the images.

So, instead I decided to add a white new patch to the images to make them 1000X1000

For eg: If the image size was 1000_750 then a white patch of remaining 1000_250 was added to the image to make it 1000X1000

```python
def correct_height(image, error):

    part_size_top = 0
    part_size_bottom = 0
    
    if error % 2 == 0:
        part_size_top = error // 2
        part_size_bottom = error // 2
    
    else:
        temp = error - 1

        part_size_top = temp // 2
        part_size_bottom = (temp // 2) + 1

    correction_part_top = np.zeros([part_size_top, image.shape[1], 3], np.uint8)
    correction_part_top.fill(255)

    correction_part_bottom = np.zeros([part_size_bottom, image.shape[1], 3], np.uint8)
    correction_part_bottom.fill(255)

    corrected_img_partial = cv2.vconcat([correction_part_top, image])
    corrected_img = cv2.vconcat([corrected_img_partial, correction_part_bottom])

    return corrected_img
```

This function corrects the image if the images if lacking pixels in terms of its height for eg: 1000X750

The same function was designed to fix the images with width issues just some changes accordingly.

Thus, this is how the images were made even sized (1000X1000)

## Corresponding Images Generation

Now, that the images were pre-processed it was time to generate the corresponding images which were required to be fed to the model with their original images.

For eg:

![new1](https://github.com/pruthvik-sheth/PlanBrainer/assets/80819203/aa72768d-c6bc-4a3f-beac-16cf8365f296)


Now, as the doors,windows as well as balconies were difficult to automate, automating boundary was quite simple and thus I did that in the first phase.

To automate boundary I used OpenCV especially the Contours API from opencv.

```python
def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    return contours

def filter_contours(contours, threshold):
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > threshold:
            filtered_contours.append(contour)
    return filtered_contours

def gen_new_image(contours):
    new_image = np.zeros([1000, 1000, 3], np.uint8)
    new_image.fill(255)
    cv2.fillPoly(new_image, contours, color=(170,170,170)) 
    return new_image

for index, image_file_name in enumerate(sorted(os.listdir(DATA_DIR))):

    try:
        image = cv2.imread(os.path.join(DATA_DIR, image_file_name))
        contours = find_contours(image)
        filtered_contours = filter_contours(contours, threshold=10000)
        generated_img = gen_new_image(filtered_contours)
        status = cv2.imwrite(os.path.join(TAR_DIR, image_file_name), generated_img)
        if status:
            print(f'Successfully generated and saved img {image_file_name}')
        else:
            print(f'Failed to generate and save img {image_file_name}')

    except Exception as e:
        print('An error occurred!:', e)
        continue
```

Output:

![[4.png]]

As far as the doors, windows and balcony positions are concerned they were manually marked by using PhotoShop and Adobe XD. Thus, a total of 925 images was reached.

## Final Dataset Generation

Now the set of images were cropped (100 pixel boundary to slightly zoom into the image) and again resized to 512X512 which suites the model architecture and were combined to be fed and used as a complete dataset.

Image Cropping and resizing code snippet

```python
cut = 100
img_size = 512

temp_data = []

for count, img in enumerate(os.listdir(DATADIR)):
    
    try:
        img_array = cv2.imread(os.path.join(DATADIR, img))
        img_height = img_array.shape[0]
        img_width = img_array.shape[1]
        # Cropping (100*100 border)
        img_array_cropped = img_array[cut:(img_width - cut), cut:(img_height - cut)]
        img_array_resized = cv2.resize(img_array_cropped, [img_size, img_size])

        filename = str(count + START_COUNT) + '.png'
        fullpath = os.path.join(DESTDIR, filename)
        # print(fullpath)

        status = cv2.imwrite(fullpath, img_array_resized)
        print(f"Image no: {count + START_COUNT} saving status: ", status)

    except Exception as e:
        print('Error: ',e)
        break
```

Image Combining Code Snippet:

```python
s_images = os.listdir(SOURCE_IMAGES_PATH)
t_images = os.listdir(TARGET_IMAGES_PATH)

for i in range(len(s_images)):

    source_img = cv2.imread(os.path.join(SOURCE_IMAGES_PATH, s_images[i]))
    target_img = cv2.imread(os.path.join(TARGET_IMAGES_PATH, t_images[i]))

    combined_img = cv2.hconcat([source_img, target_img])

    save_img(combined_img, i + START_COUNT)
```

**Output:**

![4](https://github.com/pruthvik-sheth/PlanBrainer/assets/80819203/78d5792c-8e8c-4bd7-93a7-3a94e8ab0b87)

Finally the dataset with **925** set of images was ready to be used and fed to the CGAN model.

## Training and Model Construction:

https://deepnote.com/@planbrainer/CGAN-MODEL-f68b2285-af06-4efb-83c9-0d6f39f6f2d0
