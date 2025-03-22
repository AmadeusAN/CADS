import numpy as np
import os
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
from multiprocessing import Pool
from pathlib import Path
from utils import get_pretrain_datalist, config


def truncate(CT):
    # truncate
    min_HU = -1024
    max_HU = 1024
    CT[np.where(CT <= min_HU)] = min_HU
    CT[np.where(CT >= max_HU)] = max_HU
    return CT


# rate = 1.5

spacing = {
    0: [1.0, 1.0, 1.0],
}

# ori_path = "../Images_nifti"
# save_path = "../Images_nifti_spacing"
save_path = config.cads_cache_dir

count = -1


def processing(img_path: Path):
    # img_path = os.path.join(root, i_files)  # 获取图像路径
    imageITK = sitk.ReadImage(img_path)  # 读取
    image = sitk.GetArrayFromImage(imageITK)  # 获取数组
    ori_size = np.array(imageITK.GetSize())[[2, 1, 0]]
    ori_spacing = np.array(imageITK.GetSpacing())[[2, 1, 0]]
    ori_origin = imageITK.GetOrigin()
    ori_direction = imageITK.GetDirection()

    task_id = 0  # 目标 spacing 列表，任务 0 表示 [1,1,1]
    target_spacing = np.array(spacing[task_id])
    spc_ratio = ori_spacing / target_spacing
    spc_ratio = np.round(spc_ratio, 4)

    data_type = image.dtype
    order = 3

    image = image.astype(np.float32)
    image = truncate(image)

    image_resize = resize(
        image,
        (
            int(np.round(ori_size[0] * spc_ratio[0])),
            int(ori_size[1] * spc_ratio[1]),
            int(ori_size[2] * spc_ratio[2]),
        ),
        order=order,
        cval=0,
        clip=True,
        preserve_range=True,
    )
    image_resize = np.round(image_resize).astype(
        data_type
    )  #  image_resize 为重采样后的图像数据

    # save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saveITK = sitk.GetImageFromArray(image_resize)
    saveITK.SetSpacing(target_spacing[[2, 1, 0]])
    saveITK.SetOrigin(ori_origin)
    saveITK.SetDirection(ori_direction)
    sitk.WriteImage(
        saveITK, os.path.join(save_path, img_path.name)
    )  # 对图像进行保存，并没有设置子目录


# pool = Pool(processes=8, maxtasksperchild=1000)

datalist = get_pretrain_datalist()
datalist = [Path(k["image"]) for k in datalist]

for file in tqdm(datalist):
    print(f"Processing {file.name}")
    processing(file)
    # pool.apply_async(func=processing, args=(file))
# for root, dirs, files in os.walk(ori_path):
#     for i_files in tqdm(sorted(files)):
#         if i_files[0] == ".":
#             continue

#         if os.path.isfile(os.path.join(save_path, i_files)):
#             continue

#         # read img
#         print("Processing %s" % (i_files))

#         pool.apply_async(func=processing, args=(root, i_files))

# pool.close()
# pool.join()
