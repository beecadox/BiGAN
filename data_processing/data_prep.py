# from obj to ply to h5
import os
import shutil

main_folder = "/home/beecadox/Thesis/BiGAN/data/shapenet"


def split_dataset(data):
    val = data[:100]  # 100
    test = data[100:500]  # 400
    train = data[500:]  # the rest
    return train, val, test


def copy_files():
    for subfolder in os.listdir(os.path.join(main_folder, "gt")):
        h5_model_files = os.listdir(os.path.join(main_folder, "gt", subfolder))
        train_set, val_set, test_set = split_dataset(h5_model_files)
        for train in train_set:
            if not os.path.exists(os.path.join(main_folder, "gt", "train", subfolder)):
                os.makedirs(os.path.join(main_folder, "gt", "train", subfolder))
            if not os.path.exists(os.path.join(main_folder, "partial", "train", subfolder)):
                os.makedirs(os.path.join(main_folder, "partial", "train", subfolder))

            shutil.copy(os.path.join(main_folder, "gt", subfolder, train),
                        os.path.join(main_folder, "gt", "train", subfolder))
            shutil.copy(os.path.join(main_folder, "partial", subfolder, train),
                        os.path.join(main_folder, "partial", "train", subfolder))

        for val in val_set:
            if not os.path.exists(os.path.join(main_folder, "gt", "val", subfolder)):
                os.makedirs(os.path.join(main_folder, "gt", "val", subfolder))
            if not os.path.exists(os.path.join(main_folder, "partial", "val", subfolder)):
                os.makedirs(os.path.join(main_folder, "partial", "val", subfolder))

            shutil.copy(os.path.join(main_folder, "gt", subfolder, val),
                        os.path.join(main_folder, "gt", "val", subfolder))
            shutil.copy(os.path.join(main_folder, "partial", subfolder, val),
                        os.path.join(main_folder, "partial", "val", subfolder))

        for test in test_set:
            if not os.path.exists(os.path.join(main_folder, "gt", "test", subfolder)):
                os.makedirs(os.path.join(main_folder, "gt", "test", subfolder))
            if not os.path.exists(os.path.join(main_folder, "partial", "test", subfolder)):
                os.makedirs(os.path.join(main_folder, "partial", "test", subfolder))

            shutil.copy(os.path.join(main_folder, "gt", subfolder, test),
                        os.path.join(main_folder, "gt", "test", subfolder))
            shutil.copy(os.path.join(main_folder, "partial", subfolder, test),
                        os.path.join(main_folder, "partial", "test", subfolder))


def check_gt_partial(subset):
    subset = 'test'
    equal = True
    for subfolder in os.listdir(os.path.join(main_folder, subset, "gt")):
        gt_files = os.listdir(os.path.join(main_folder, subset, "gt", subfolder))
        partial_files = os.listdir(os.path.join(main_folder, subset, "partial", subfolder))
        if gt_files != partial_files:
            equal = False
        else:
            with open("/home/beecadox/Thesis/BiGAN/data/shapenet/test.list", 'a') as f:
                for listitem in gt_files:
                    f.write(subfolder + "/" + listitem.split(".h5")[0] + "\n")
    if equal:
        print(subset, " ok")
    else:
        print(subset, " not ok")


# copy_files()
check_gt_partial("train")
check_gt_partial("val")
check_gt_partial("test")

