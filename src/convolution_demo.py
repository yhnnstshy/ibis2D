from csv import reader as csv_reader
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage.morphology import disk
from skimage.draw import line as draw_line
from skimage.segmentation import watershed
from scipy.ndimage.filters import convolve
from scipy.ndimage import label
from PIL import Image
from skimage.color import rgb2gray


source = 'C:\Users\Andrei\Dropbox\Andrei\Data_Final\PyMT\XY\mouse_2_Day5'
source_2 = 'C:\Users\Andrei\Dropbox\Andrei\Data_Final\PyMT\Images\mouse_2_Day5'

for fle in os.listdir(source):
    print os.path.join(source, fle)
    with open(os.path.join(source, fle)) as coordinates_source:
        reader = csv_reader(coordinates_source, delimiter='\t')
        xy = []
        first_line = reader.next()
        xy.append(first_line)
        for line in reader:
            xy.append(line)
        xy.append(first_line)

        xy = np.array(xy).astype(np.float)*2 - 16
        xy = xy.astype(np.int)

        base_array = np.zeros((1388, 1040))

        for i in range(0, xy.shape[0]-1):
            rr, cc = draw_line(xy[i, 0], xy[i, 1], xy[i+1, 0], xy[i+1, 1])
            base_array[rr, cc] = 1

        im = Image.open(os.path.join(source_2, fle)[:-3]+'tif')
        im = np.array(im)
        # plt.imshow(im)
        # plt.show()

        dic = rgb2gray(im[:, :1388])
        k14 = rgb2gray(im[:, 1388:])

        plt.imshow(dic, cmap='Greys')
        plt.axis('off')
        plt.savefig("DIC.png", bbox_inches='tight')
        # plt.savefig("test.png", bbox_inches='tight')

        plt.imshow(k14, cmap='Greens')
        plt.axis('off')
        # plt.show()
        plt.savefig("K14.png", bbox_inches='tight')

        plt.imshow(dic, cmap='Greys')
        plt.imshow(k14, cmap='Greens', alpha=0.5)

        plt.imshow(dic, cmap='Greys')
        plt.imshow(k14, cmap='Greens', alpha=0.5)

        plt.plot(xy[:, 0], xy[:, 1])
        # plt.imshow(base_array.T, alpha=0.3, cmap='Reds')
        plt.axis('off')
        # plt.show()
        plt.savefig("Border.png", bbox_inches='tight')

        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]

        index_array, nf = label(1-base_array)
        background = index_array[0, 0]
        index_array[index_array == background] = 0

        plt.imshow(dic, cmap='Greys')
        plt.imshow(index_array.T, cmap='Reds', alpha=0.5)
        # plt.colorbar()
        plt.axis('off')
        # plt.show()
        plt.savefig("internal_map.png", bbox_inches='tight')

        element = disk(20)

        conv_array = convolve(index_array, element)

        conv_array = conv_array/float(conv_array.max())

        plt.imshow(conv_array.T, cmap='Reds')
        plt.colorbar()
        plt.axis('off')
        # plt.show()
        plt.savefig("convolved_array.png", bbox_inches='tight')

        plt.imshow(dic, cmap='Greys')
        plt.imshow(np.logical_and(conv_array.T > 0.5, conv_array.T < 0.99), cmap='Reds', alpha=0.5)
        # plt.plot(xy[:, 0], xy[:, 1])
        plt.axis('off')
        # plt.show()
        plt.savefig("inner_border.png", bbox_inches='tight')

        plt.imshow(dic, cmap='Greys')
        plt.imshow(conv_array.T > 0.99, cmap='Reds', alpha=0.5)
        # plt.plot(xy[:, 0], xy[:, 1])
        plt.axis('off')
        # plt.show()
        plt.savefig("internal_area.png", bbox_inches='tight')

        raw_input('Press enter to continue')