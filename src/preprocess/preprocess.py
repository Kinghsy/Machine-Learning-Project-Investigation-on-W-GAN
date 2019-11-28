import os
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io

import numpy as np
from tqdm import tqdm


extension = ".gif"

fixed_size = 64
margin = 10

def main():

    img_msk_path = os.path.join("..", '..', 'data', 'data_raw', 'train_masks')
    img_path = os.path.join("..", '..', 'data', 'data_raw', 'train')
    img_target_path = os.path.join("..", '..', 'data', '64x64')
    files = os.listdir(img_path)

    for file in tqdm(files):
        img_file = os.path.join(img_path, file)
        img_msk_file = os.path.join(img_msk_path, file.split(".")[0]+"_mask"+extension)

        img = io.imread(img_file)
        shp = img.shape
        img_msk = io.imread(img_msk_file)
        img_msk = (img_msk > 0)

        img_full_msk = np.zeros_like(img)
        img_full_msk[:, :, 0] = img_msk
        img_full_msk[:, :, 1] = img_msk
        img_full_msk[:, :, 2] = img_msk
        img_blank = np.ones_like(img) * 255

        img_msked = img * img_full_msk + img_blank * (1 - img_full_msk)

        img_col = np.sum(img_msk, axis=1)
        img_row = np.sum(img_msk, axis=0)

        l = [0, 0]
        r = [0, 0]

        for i, val in enumerate(img_col):
            if val>0:
                if l[0] == 0:
                    l[0] = i
                r[0] = i

        for i, val in enumerate(img_row):
            if val>0:
                if l[1] == 0:
                    l[1] = i
                r[1] = i

        mid = [(l[0]+r[0])//2, (l[1]+r[1])//2]

        rag = max(mid[0]-l[0], r[0]-mid[0], mid[1]-l[1], r[1]-mid[1])
        l_rag = [mid[0]-rag-margin, mid[1]-rag-margin]
        r_rag = [mid[0]+rag+margin, mid[1]+rag+margin]

        img_cut = np.zeros((r_rag[0]-l_rag[0], r_rag[1]-l_rag[1], 3))
        l_bound = [max(l_rag[0], 0), max(l_rag[1], 0)]
        r_bound = [min(r_rag[0], shp[0]), min(r_rag[1], shp[1])]
        # img_cut[0:r_bound[0]-l_bound[0], 0:r_bound[1]-l_bound[1], :] = img_msked[l_bound[0]:r_bound[0], l_bound[1]:r_bound[1], :]
        # img_cut = img_msked[l_rag[0]:r_rag[0], l_rag[1]:r_rag[1], :]
        img_resized = resize(img_msked[l_bound[0]:r_bound[0], l_bound[1]:r_bound[1], :], (fixed_size, fixed_size), anti_aliasing=True)

        # plt.imshow(img_resized)
        plt.imsave(os.path.join(img_target_path, file), img_resized)

        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
        #
        # ax = axes.ravel()
        #
        # ax[0].imshow(img)
        # ax[0].set_title("Original image")
        #
        # ax[1].imshow(img_resized)
        # ax[1].set_title("Resized image (no aliasing)")
        #
        # plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    main()
