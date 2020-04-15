import os
import sys
from skimage import io
from PIL import Image, ImageOps
from pathlib import Path
from scipy import stats
import numpy as np
import csv


class Colony:
    def __init__(self, img_path: str, img: np.ndarray, ant_count: int, pheromone_evaporation_constant=0.1,
                 pheromone_memory_constant=20, ant_memory_constant=100, minimum_pheromone_constant=0.0001,
                 intensity_threshold_value=0.05, alpha=1.0, beta=1.0) -> None:
        self.img_path = img_path
        self.img = img
        # self.i_max = img.max()
        self.intensities = np.empty(shape=(self.img.shape[0], self.img.shape[1]), dtype=np.dtype(np.uint8))
        self.set_pixel_intensities(normalize=False)
        self.generate_intensities_image(invert=True, binary=True)

    def pixel_intensity(self, row: int, col: int) -> float:
        """
        Caclulates the intensity/importance of a given pixel by looking at surrounding pixels
        :param row: x index for the pixel
        :param col: y index for the pixel
        :return: Normalized maximum intensity
        """
        # (1 / self.i_max)
        return max(
            abs(int(self.img[row - 3, col - 3]) - int(self.img[row + 3, col + 3])) if (row - 3 >= 0 and col - 3 >= 0 and row + 3 < self.img.shape[0] and col + 3 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 3, col - 2]) - int(self.img[row + 3, col + 2])) if (row - 3 >= 0 and col - 2 >= 0 and row + 3 < self.img.shape[0] and col + 2 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 3, col - 1]) - int(self.img[row + 3, col + 1])) if (row - 3 >= 0 and col - 1 >= 0 and row + 3 < self.img.shape[0] and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 3, col    ]) - int(self.img[row + 3, col    ])) if (row - 3 >= 0 and row + 3 < self.img.shape[0]) else 0,
            abs(int(self.img[row - 3, col + 1]) - int(self.img[row + 3, col - 1])) if (row - 3 >= 0 and col - 1 >= 0 and row + 3 < self.img.shape[0] and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 3, col + 2]) - int(self.img[row + 3, col - 2])) if (row - 3 >= 0 and col - 2 >= 0 and row + 3 < self.img.shape[0] and col + 2 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 3, col + 3]) - int(self.img[row + 3, col - 3])) if (row - 3 >= 0 and col - 3 >= 0 and row + 3 < self.img.shape[0] and col + 3 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 2, col + 3]) - int(self.img[row + 2, col - 3])) if (row - 2 >= 0 and col - 3 >= 0 and row + 2 < self.img.shape[0] and col + 3 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 1, col + 3]) - int(self.img[row + 1, col - 3])) if (row - 1 >= 0 and col - 3 >= 0 and row + 1 < self.img.shape[0] and col + 3 < self.img.shape[1]) else 0,
            abs(int(self.img[row    , col + 3]) - int(self.img[row    , col - 3])) if (col - 3 >= 0 and col + 3 < self.img.shape[1]) else 0,
            abs(int(self.img[row + 1, col + 3]) - int(self.img[row - 1, col - 3])) if (row - 1 >= 0 and col - 3 >= 0 and row + 1 < self.img.shape[0] and col + 3 < self.img.shape[1]) else 0,
            abs(int(self.img[row + 2, col + 3]) - int(self.img[row - 2, col - 3])) if (row - 2 >= 0 and col - 3 >= 0 and row + 2 < self.img.shape[0] and col + 3 < self.img.shape[1]) else 0,

            abs(int(self.img[row - 2, col - 2]) - int(self.img[row + 2, col + 2])) if (row - 2 >= 0 and col - 2 >= 0 and row + 2 < self.img.shape[0] and col + 2 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 2, col - 1]) - int(self.img[row + 2, col + 1])) if (row - 2 >= 0 and col - 1 >= 0 and row + 2 < self.img.shape[0] and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 2, col    ]) - int(self.img[row + 2, col    ])) if (row - 2 >= 0 and row + 2 < self.img.shape[0]) else 0,
            abs(int(self.img[row - 2, col + 1]) - int(self.img[row + 2, col - 1])) if (row - 2 >= 0 and col - 1 >= 0 and row + 2 < self.img.shape[0] and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 2, col + 2]) - int(self.img[row + 2, col - 2])) if (row - 2 >= 0 and col - 2 >= 0 and row + 2 < self.img.shape[0] and col + 2 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 1, col + 2]) - int(self.img[row + 1, col - 2])) if (row - 1 >= 0 and col - 2 >= 0 and row + 1 < self.img.shape[0] and col + 2 < self.img.shape[1]) else 0,
            abs(int(self.img[row    , col + 2]) - int(self.img[row    , col - 2])) if (col - 2 >= 0 and col + 2 < self.img.shape[1]) else 0,
            abs(int(self.img[row + 1, col + 2]) - int(self.img[row - 1, col - 2])) if (row - 1 >= 0 and col - 2 >= 0 and row + 1 < self.img.shape[0] and col + 2 < self.img.shape[1]) else 0,

            abs(int(self.img[row - 1, col - 1]) - int(self.img[row + 1, col + 1])) if (row - 1 >= 0 and col - 1 >= 0 and row + 1 < self.img.shape[0] and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 1, col + 1]) - int(self.img[row + 1, col - 1])) if (row - 1 >= 0 and col - 1 >= 0 and row + 1 < self.img.shape[0] and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row    , col - 1]) - int(self.img[row    , col + 1])) if (col - 1 >= 0 and col + 1 < self.img.shape[1]) else 0,
            abs(int(self.img[row - 1, col    ]) - int(self.img[row + 1, col    ])) if (row - 1 >= 0 and row + 1 < self.img.shape[0]) else 0
        )

    def perform_max_normalization_intensities(self):
        max_val = self.intensities.max()
        for i, j in np.ndindex(self.intensities.shape):
            self.intensities[i, j] /= max_val
        return self.intensities

    def normalize_intensities(self, zscore=True):
        return stats.zscore(self.intensities) if zscore else self.perform_max_normalization_intensities()

    def set_pixel_intensities(self, normalize=True):
        """
        Creates and stores all of the pixel intensities, no need to keep recomputing since the value never changes
        :return: Nothing
        """
        print("Image shape for pixel intensities: " + str(self.img.shape))
        for i, j in np.ndindex(self.img.shape):
            self.intensities[i, j] = self.pixel_intensity(i, j)
        if (normalize == True):
            self.intensities = self.normalize_intensities(zscore=False)
        print("Intensity: max: " + str(self.intensities.max()) + " min: " + str(self.intensities.min()) +
              " average intensity: " + str(self.intensities.mean()))
        print(self.intensities)
        # Image.fromarray(self.intensities, 'L').show()

    def generate_intensities_image(self, invert=True, binary=True):
        """
        Creates and stores the intensities matrix as a normal gray-scale image
        :return: Nothing
        """
        base_dir = os.path.dirname(self.img_path)
        intensities_path = os.path.join(base_dir, "Intensities-test")

        # Path.mkdir(intensities_path, parents=True, exist_ok=True)

        base = os.path.basename(self.img_path)
        # adjusted_path = os.path.join(os.path.dirname(self.img_path), "intensities_", base)
        final_path = os.path.join(intensities_path, base)
        arr = self.convert_to_gray(self.intensities, binary=binary)
        generate_image_from_array(path=final_path, array=arr, invert=invert)
        # Image.fromarray(self.intensities, 'L').save(final_path)

    # @staticmethod
    def convert_to_gray(self, arr, binary=True):
        """
        Converts a given 2d ndarray to gray-scaling
        :param arr: 2d ndarray
        :return: arr
        """
        arr = arr.copy()
        old_max = arr.max()
        old_min = arr.min()
        old_avg = arr.mean()
        print("Old min: " + str(old_min) + " old max: " + str(old_max) + " range: " + str(old_max - old_min) +
              " average: " + str(old_avg))
        for i, j in np.ndindex(arr.shape):
            if (binary == True):
                if (arr[i, j] >= old_avg):
                    arr[i, j] = 255
                else:
                    arr[i, j] = 0
            else:
                arr[i, j] = int((((255 - 0) * (arr[i, j] - old_min)) / (old_max - old_min)) + 0 + 0.5)
        return arr.astype(dtype=np.dtype(np.uint8), casting='safe', copy=True)

    def print_intensities(self):
        base_dir = os.path.dirname(self.img_path)
        intensities_path = os.path.join(base_dir, "Intensities-test")

        # Path.mkdir(intensities_path, parents=True, exist_ok=True)

        base = os.path.basename(self.img_path)
        base = base.split(sep='.')
        base = ''.join(base[:-1]) + ".csv"
        final_path = os.path.join(intensities_path, base)
        print("Intensities: type: " + str(self.intensities.dtype) + " max: " + str(self.intensities.max()) + " min: " +
              str(self.intensities.min()) + " average: " + str(self.intensities.mean()))
        with open(final_path, 'w') as csvfile:
            current_i = 0
            for i, j in np.ndindex(self.intensities.shape):
                if (i > current_i):
                    current_i += 1
                    print(file=csvfile)
                # print("(" + str(i) + ", " + str(j) + ") [" + str(self.intensities[i, j]) + "]")
                # print(str(self.intensities[i, j]), sep=',')
                print(str(self.intensities[i, j]), end=',', file=csvfile)
            print(file=csvfile)

    def clean_up(self, dir_path):
        """
        Clears the directory dir_path
        :param dir_path: directory to clear
        :return: Nothing
        """
        print("cleaning " + dir_path + " ...")
        entries = None
        try:
            entries = os.listdir(path=dir_path)
        except FileNotFoundError as FNFE:
            print(type(FNFE).__name__ + ": " + argv[0] + ": " + str(FNFE), file=sys.stderr)
            return
        for entry in entries:
            path = os.path.join(dir_path, entry)
            try:
                os.remove(path=path)
            except FileNotFoundError:
                break
        print("Done cleaning!")


def generate_image_from_array (path, array: np.ndarray, invert=True):
    """
    Inverts the array scheme and physically saves it
    :param path: directory to store
    :param array: 2d ndarray of data
    :return: Nothing
    """
    dir_base = os.path.dirname(path)
    Path(dir_base).mkdir(parents=True, exist_ok=True)
    print("array: type: " + str(array.dtype))
    img = Image.fromarray(array, 'L')
    if (invert == True):
        img = ImageOps.invert(img)
    print(img)
    img.save(path)


if __name__ == "__main__":
    argv = sys.argv
    print("Args: " + str(argv))
    if (len(argv) != 2):
        print("Usage: " + argv[0] + ": `" + os.path.basename(argv[0]) + " <image directory>'", file=sys.stderr)
        exit(1)

    entries = None
    try:
        entries = os.listdir(path=argv[1])
    except FileNotFoundError as FNFE:
        print(type(FNFE).__name__ + ": " + argv[0] + ": " + str(FNFE), file=sys.stderr)
        exit(2)

    print("Directory [" + argv[1] + "]: " + str(entries))

    for item in entries:
        if (item == "Intensities" or item == "Iterations" or item == "Intensities-test"):
            continue
        path = os.path.join(argv[1], item)
        try:
            img = io.imread(path)
            print("Read in: `" + path + "'")
        except ValueError as VE:
            print(type(VE).__name__ + ": " + argv[0] + ": `" + path + "' is not a valid image", file=sys.stderr)
            continue
        print("[" + item + "] size: " + str(img.shape) + " len: " + str(img.shape[0] * img.shape[1]))
        print(img)
        print("Image: type: " + str(img.dtype) + " max: " + str(img.max()) + " min: " + str(img.min()) + " mean: " + str(img.mean()))
        dir = os.path.dirname(argv[1])
        fname = item
        fname = fname.split(sep='.')[:-1]
        fname.append('.csv')
        fname = ''.join(fname)
        final_path = os.path.join(dir, "Intensities-test", fname)
        with open(final_path, 'w') as csvfile:
            current_i = 0
            for i, j in np.ndindex(img.shape):
                if (i > current_i):
                    current_i += 1
                    print(file=csvfile)
                # print("(" + str(i) + ", " + str(j) + ") [" + str(self.intensities[i, j]) + "]")
                # print(str(self.intensities[i, j]), sep=',')
                print(str(img[i, j]), end=',', file=csvfile)
            print(file=csvfile)
        c = Colony(img_path=path, img=img, ant_count=750, pheromone_evaporation_constant=0.001,
                   pheromone_memory_constant=30, ant_memory_constant=30, intensity_threshold_value=0.0967,
                   alpha=2.5, beta=2.0)
        c.print_intensities()
        break
        # clean_path = os.path.join(argv[1], "Iterations", item.split('.')[0])
        # c.clean_up(dir_path=clean_path)
        # c.iterate(1000)
        # c.adjust_pheromone()
