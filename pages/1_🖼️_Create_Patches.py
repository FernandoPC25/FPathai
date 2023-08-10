OPENSLIDE_PATH = r'c:\Users\Fernando\Desktop\MASTER UGR\4-TFM\openslide-win64\bin'
import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import os

import h5py
import numpy as np
import streamlit as st
from openslide import OpenSlide
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.color import rgb2hsv
from skimage.exposure.exposure import is_low_contrast
from skimage.filters import threshold_otsu


def get_mask_image(img_RGB, RGB_min=50):
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask


def get_mask(slide, level='max', RGB_min=50):
    # read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level


def extract_patches(slide_path, mask_path, patch_size, patches_output_dir, slide_id, max_patches_per_slide=2000):
    patch_folder = os.path.join(patches_output_dir, slide_id)
    print("patch_folder: ", patch_folder)
    if not os.path.isdir(patch_folder):
        os.makedirs(patch_folder)
    else:
        return
    try:
        slide = OpenSlide(slide_path)
    except:
        return

    patch_folder_mask = os.path.join(mask_path, slide_id)
    if not os.path.isdir(patch_folder_mask):
        os.makedirs(patch_folder_mask)
        mask, mask_level = get_mask(slide)
        mask = binary_dilation(mask, iterations=3)
        mask = binary_erosion(mask, iterations=3)
        np.save(os.path.join(patch_folder_mask, "mask.npy"), mask)
        print("path mask", os.path.join(mask_path, slide_id, 'mask.npy'))
    else:
        mask = np.load(os.path.join(mask_path, slide_id, 'mask.npy'))

    mask_level = len(slide.level_dimensions) - 1

    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    try:
        ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
        print("ratio_x: ", ratio_x)
        ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]
        print("ratio_y: ", ratio_y)

        xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

        # handle slides with 40 magnification at base level
        resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
        print("resize_factor: ", resize_factor)
        # resize_factor = resize_factor * args.dezoom_factor
        print("patch_size[0]: ", patch_size[0])
        patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
        print("patch_size_resized: ", patch_size_resized)
        i = 0

        indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                   range(0, ymax, patch_size_resized[0])]
        # print("indices: ", indices)
        # np.random.seed(5)
        # np.random.shuffle(indices)
        path_h5 = os.path.join(patch_folder, slide_id + '.h5')
        print("path_h5: ", path_h5)
        if os.path.exists(path_h5):
            print('Image already converted')
            return
        with h5py.File(path_h5, 'w') as f:
            for x, y in indices:
                # check if in background mask
                x_mask = int(x / ratio_x)
                # print("x_mask: ", x_mask)
                y_mask = int(y / ratio_y)
                # print("y_mask: ", y_mask)
                # print("mask[x_mask, y_mask]: ", mask[x_mask, y_mask])
                if mask[x_mask, y_mask] == 1:
                    patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')
                    # print("patch: ", patch)
                    try:
                        mask_patch = get_mask_image(np.array(patch))
                        mask_patch = binary_dilation(mask_patch, iterations=3)
                    except Exception as e:
                        print("error with slide id {} patch {}".format(slide_id, i))
                        print(e)
                    if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                        if resize_factor != 1.0:
                            patch = patch.resize(patch_size)
                        img_idx = str(i)
                        patch = np.array(patch)
                        dset = f.create_dataset(img_idx, data=patch)
                        i += 1

                if i >= max_patches_per_slide:
                    break

        if i == 0:
            print("no patch extracted for slide {}".format(slide_id))

    except Exception as e:
        print("error with slide id {} patch {}".format(slide_id, i))
        print(e)


def get_slide_id(slide_name):
    return slide_name.split('.')[0] + '.' + slide_name.split('.')[1]


st.set_page_config(
    page_title="Create patches",
)
st.title("Create patches")
st.write("Lorem ipsum")

# with st.sidebar:
#     decision = st.radio(
#         "Do you need to create a CSV?",
#         ('Yes', 'No'))
#     if decision == 'Yes':
#         st.title("Create CSV")
#     else:
#         st.write("Then upload the CSV to start the training.")

def svs_images():
    """
    Generate the csv
    """
    directory = st.text_input("Path to the svs folder:")
    if not directory:
        st.info("Please specify the directory containing the .svs format images for which you wish to generate patches.", icon="ℹ")
        return
    if not os.path.isdir(directory):
        st.error("Invalid directory path.", icon="🚨")
        return

    svs_images = [filename for filename in os.listdir(directory) if filename.lower().endswith(".svs")]

    if not svs_images:
        st.info("No .svs images found in the directory.")
        return
    st.success(f"Found {len(svs_images)} .svs image(s) in the directory.")

    for svs_image in svs_images:
        full_path = os.path.join(directory, svs_image)
        #openslide.open_slide(full_path).get_thumbnail(size=(1024, 1024))
        st.image(openslide.open_slide(full_path).get_thumbnail(size=(1024, 1024)), caption=f"{svs_image}")

    mask_path = "mask"
    patch_path = "patches"
    mask_path = os.path.join(directory, mask_path)
    patch_path = os.path.join(directory, patch_path)
    st.success(f"The patches for these images are going to be created in {patch_path}")

    with st.form("my_form"):
        st.write("Select a patch size")
        patch = st.slider('Choose the patch for your images', 0, 1024, 256)

        submitted = st.form_submit_button("Create patches")
        if submitted:
            st.write(f"You have selected a patch size of {patch,patch}")
            patch_size = (patch, patch)

            progress_bar = st.progress(0)
            total_images = len(svs_images)
            for (i, s) in enumerate(svs_images):
                slide_path = os.path.join(directory, svs_images[i])
                slide_id = get_slide_id(svs_images[i])
                progress = (i + 1) / total_images
                progress_bar.progress(progress, text=f"Creating patches of {slide_id}... Please wait")
                extract_patches(slide_path, mask_path, patch_size, patch_path, slide_id)
            st.success("Patches created")


svs_images()

# # Ruta a la imagen SVS
# svs_image_path = r'c:\Users\Fernando\Desktop\MASTER UGR\4-TFM\TFM-streamlit\svs_a_h5\control_svs\TCGA-22-4595-11A-01-BS1.460293a1-334c-4a57-999d-a9ba82fb289b.svs'
#
# svs_image_path_2 = r'c:\Users\Fernando\Desktop\MASTER UGR\4-TFM\TFM-streamlit\svs_a_h5\control_svs\TCGA-22-5472-11A-01-TS1.06b7b21e-715d-43c6-816c-605018802020.svs'
#
#
# # Cargar la imagen SVS usando OpenSlide
# slide = openslide.open_slide(svs_image_path)
#
# slide2 = openslide.open_slide(svs_image_path_2)
#
#
# img = slide.get_thumbnail(size=(1024, 1024))
#
# img2 = slide2.get_thumbnail(size=(1024, 1024))
#
#
# # # Convertir la imagen a formato PNG
# # level = 2
# # imagen= slide.read_region((0, 0), level, slide.level_dimensions[level])
#
# # Mostrar la imagen PNG convertida usando st.image()
# st.image(img, caption='Imagen SVS convertida a PNG', use_column_width="always")
#
# st.image(img2, caption='Imagen SVS convertida a PNG', use_column_width="always")