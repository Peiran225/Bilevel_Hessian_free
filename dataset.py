from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode
from torchmeta.utils.data import BatchMetaDataLoader
# from    PIL import Image
import  os.path
import  numpy as np
import torch

class OmniglotNShot:

    def __init__(self, batchsz, n_way, k_shot, k_query, imgsz=28, spider_batchsz=0):
        """
        Different from mnistNShot, the
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz

        prefix_dir =  '/ocean/projects/cis220038p/junyili/AdaBilevel'

        dataset = Omniglot(prefix_dir + "/processed_data",
                        # Number of ways
                        num_classes_per_task=n_way,
                        # Resize the images to 28x28 and converts them\
                        #  to PyTorch tensors (from Torchvision)
                        transform=Compose([Resize(imgsz, InterpolationMode.LANCZOS), ToTensor()]),
                        # Transform the labels to integers (e.g.\
                        #  ("Glagolitic/character01", "Sanskrit/character14", ...) \
                        # to (0, 1, ...))
                        target_transform=Categorical(num_classes=n_way),
                        # Creates new virtual classes with rotated versions \
                        # of the images (from Santoro et al., 2016)
                        # class_augmentations=[Rotation([30, 90, 120, 180, 270])],
                        class_augmentations=[Rotation([90, 180, 270])],
                        meta_train=True, download=True)
          
        dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=k_shot, num_test_per_class=k_query)
        self.dataloader = BatchMetaDataLoader(dataset, batch_size=batchsz, num_workers=4)
        if spider_batchsz > 0:
            self.spider_dataloader = BatchMetaDataLoader(dataset, batch_size=spider_batchsz, num_workers=4)


        dataset_val = Omniglot(prefix_dir + "/processed_data",
                        # Number of ways
                        num_classes_per_task=n_way,
                        # Resize the images to 28x28 and converts them\
                        #  to PyTorch tensors (from Torchvision)
                        transform=Compose([Resize(imgsz,InterpolationMode.LANCZOS), ToTensor()]),
                        # Transform the labels to integers (e.g.\
                        #  ("Glagolitic/character01", "Sanskrit/character14", ...) \
                        # to (0, 1, ...))
                        target_transform=Categorical(num_classes=n_way),
                        # Creates new virtual classes with rotated versions \
                        # of the images (from Santoro et al., 2016)
                        # class_augmentations=[Rotation([30, 90, 120, 180, 270])],
                        class_augmentations=[Rotation([90, 180, 270])],
                        meta_val=True, download=False)

        dataset_val = ClassSplitter(dataset_val, shuffle=True, num_train_per_class=k_shot, num_test_per_class=15)
        self.dataloader_val = BatchMetaDataLoader(dataset_val, shuffle=True, batch_size=batchsz, num_workers=4)

class MiniImagenetNShot:

    def __init__(self, batchsz, n_way, k_shot, k_query, imgsz=84):
        """
        Different from mnistNShot, the
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz

        prefix_dir =  '/ocean/projects/cis220038p/junyili/AdaBilevel'

        dataset = MiniImagenet(prefix_dir + "/processed_data",
                        # Number of ways
                        num_classes_per_task=n_way,
                        # Resize the images to 28x28 and converts them\
                        #  to PyTorch tensors (from Torchvision)
                        transform=Compose([Resize(imgsz,InterpolationMode.BICUBIC), ToTensor()]),
                        # Transform the labels to integers (e.g.\
                        #  ("Glagolitic/character01", "Sanskrit/character14", ...) \
                        # to (0, 1, ...))
                        target_transform=Categorical(num_classes=n_way),
                        # Creates new virtual classes with rotated versions \
                        # of the images (from Santoro et al., 2016)
                        # class_augmentations=[Rotation([30, 90, 120, 180, 270])],
                        # class_augmentations=[Rotation([90, 180, 270])],
                        meta_train=True, download=True)

        dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=k_shot, num_test_per_class=k_query)
        self.dataloader = BatchMetaDataLoader(dataset, batch_size=batchsz, num_workers=4)

        dataset_val = MiniImagenet(prefix_dir +  "/processed_data",
                        # Number of ways
                        num_classes_per_task=n_way,
                        # Resize the images to 28x28 and converts them\
                        #  to PyTorch tensors (from Torchvision)
                        transform=Compose([Resize(imgsz,InterpolationMode.BICUBIC), ToTensor()]),
                        # Transform the labels to integers (e.g.\
                        #  ("Glagolitic/character01", "Sanskrit/character14", ...) \
                        # to (0, 1, ...))
                        target_transform=Categorical(num_classes=n_way),
                        # Creates new virtual classes with rotated versions \
                        # of the images (from Santoro et al., 2016)
                        # class_augmentations=[Rotation([30, 90, 120, 180, 270])],
                        # class_augmentations=[Rotation([90, 180, 270])],
                        meta_val=True, download=False)

        dataset_val = ClassSplitter(dataset_val, shuffle=True, num_train_per_class=k_shot, num_test_per_class=15)
        self.dataloader_val = BatchMetaDataLoader(dataset_val, shuffle=True, batch_size=batchsz, num_workers=4)


if __name__ == '__main__':
    MiniImagenetNShot(4, 5, 1, 15)