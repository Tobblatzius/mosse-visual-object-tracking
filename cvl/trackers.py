import numpy as np
from copy import copy
# from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch
from cvl.utils import *

class NCCTracker:

    def __init__(self, learning_rate=0.1):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

    def crop_patch(self, image):
        region = self.region
        return crop_patch(image, region)

    def start(self, image, region, original_image_shape=None, deep_image_shape=None):
        image = image.squeeze()
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.crop_patch(image)

        patch = patch/255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)

        self.template = fft2(patch)

    def detect(self, image):
        image = image.squeeze()
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)

        responsef = np.conj(self.template) * patchf
        response = ifft2(responsef)

        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.prev_filter_region = copy(self.region)

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

    def update(self, image, lr=0.1):
        image = image.squeeze()
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        try:
            patch = self.crop_patch(image)
        except:
            self.region = self.prev_filter_region
            patch = self.crop_patch(image)

        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        patchf = fft2(patch)
        self.template = self.template * (1 - lr) + patchf * lr


class MOSSE:
    def __init__(self, learning_rate=0.02, sigma=100, filter_region_scale=1, r_search=0, deep_features=False):
        # lr 0.02
        # sigma 10
        # filter_region_scale is a scaling factor. A value of 1 means that the boundary box is used as the region of the image that the filter will be ran over. A value of 2 means that we search in a region twice as wide and high as the boundary box but with the same center.
        self.filter_region_scale = filter_region_scale
        self.r_search = r_search
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate
        self.deep_features = deep_features
        self.sigma = sigma

    def set_filter_region(self):

        if self.deep_features:
            self.filter_region.xpos = int(self.deep_region.xpos - (self.filter_region_scale-1)*self.deep_region.width/2)
            self.filter_region.ypos = int(self.deep_region.ypos - (self.filter_region_scale-1)*self.deep_region.height/2)
        else:
            self.filter_region.xpos = int(self.region.xpos - (self.filter_region_scale-1)*self.region.width/2)
            self.filter_region.ypos = int(self.region.ypos - (self.filter_region_scale-1)*self.region.height/2)

    def set_region(self):
        if self.deep_features:
            self.region.xpos = int((self.filter_region.xpos + (self.filter_region_scale-1)*self.deep_region.width/2)/self.width_scale) 
            self.region.ypos = int((self.filter_region.ypos + (self.filter_region_scale-1)*self.deep_region.height/2)/self.height_scale)
        else: 
            self.region.xpos = int(self.filter_region.xpos + (self.filter_region_scale-1)*self.region.width/2)
            self.region.ypos = int(self.filter_region.ypos + (self.filter_region_scale-1)*self.region.height/2)


    def set_deep_region(self, deep_image_shape, original_image_shape):
        height, width, channels = original_image_shape
        deep_height, deep_width, deep_channels = deep_image_shape
        self.height_scale = deep_height/height
        self.width_scale = deep_width/width

        self.deep_region = copy(self.region)
        # floor since in upper left corner.
        self.deep_region.ypos = int(self.deep_region.ypos*self.height_scale)
        self.deep_region.xpos = int(self.deep_region.xpos*self.width_scale)
        self.deep_region.height = int(np.ceil(self.deep_region.height*self.height_scale))
        self.deep_region.width = int(np.ceil(self.deep_region.width*self.width_scale))
        self.deep_region_shape = (self.deep_region.height, self.deep_region.width)
        self.deep_region_center = (int(self.deep_region.height // 2), int(self.deep_region.width // 2))

    def start(self, image, region, deep_image_shape=None, original_image_shape=None):

        image = np.transpose(image, axes=[2, 0, 1])       
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)

        if self.deep_features:
            self.set_deep_region(deep_image_shape, original_image_shape)
            self.filter_region = copy(self.deep_region)
        else:
            self.filter_region = copy(self.region)

        self.filter_region.width = np.int(np.ceil(self.filter_region.width*self.filter_region_scale))
        self.filter_region.height = np.int(np.ceil(self.filter_region.height*self.filter_region_scale))
        self.set_filter_region()

        assert self.filter_region.width > 1 and self.filter_region.height > 1, "the calculated filter_region is too small to be used with MOSSE."


        if min(self.filter_region.width, self.filter_region.height) < self.r_search:
            print('Specified r_search is larger than the filter region, setting r_search to proper size.')
            self.r_search = min(self.filter_region.width, self.filter_region.height)

        fi = np.array([crop_patch(c, self.filter_region) for c in image])
        g = crop_patch(self._get_gauss_response(image, self.filter_region), self.filter_region)
        self.G = fft2(g)
        fi = pre_process(fi)
        Fi = fft2(fi)
        Fi_conj = np.conjugate(Fi)
        self.Ai = self.G * Fi_conj
        self.Bi = Fi * Fi_conj


    def detect(self, image):
        image = np.transpose(image, axes=[2, 0, 1])
        Hi = self.Ai / (self.Bi)
        fi = np.array([crop_patch(c, self.filter_region) for c in image])
        Fi = fft2(fi)
        Gi = Hi * Fi
        gi = ifft2(Gi).real
        # sum over channels
        gi = np.sum(gi, axis=0)
        # divide height or width by 2 and scale by search_from_center factor to set how many rows and cols from the current center we are looking for a new center.
        if self.r_search==0:
            max_value = np.max(gi)
        else:
            from_center_row = int(np.ceil(self.r_search/2))
            from_center_col = int(np.ceil(self.r_search/2))
            row_fr = int(gi.shape[0]/2 - from_center_row)
            col_fr = int(gi.shape[1]/2 - from_center_col)
            max_value = np.max(gi[row_fr:row_fr+2*from_center_row, col_fr:col_fr+2*from_center_col])

        max_pos = np.where(gi == max_value)
        try:
            dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
            dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
        except:
            dx = 0
            dy = 0
        self.prev_filter_region = copy(self.filter_region)

        self.filter_region.xpos += dx
        self.filter_region.ypos += dy
        self.set_region()

    def update(self, image):
        image = np.transpose(image, axes=[2, 0, 1])
        try:       
            fi = np.array([crop_patch(c, self.filter_region) for c in image])
        except:
            print('Last filter was outside of image and the crop failed, using the most recent filter instead.')
            self.filter_region = self.prev_filter_region
            fi = np.array([crop_patch(c, self.filter_region) for c in image])
        
        Fi = fft2(fi)
        Fi_conj = np.conjugate(Fi)

        self.Ai = self.learning_rate * self.G * Fi_conj + (1 - self.learning_rate) * self.Ai
        self.Bi = self.learning_rate * Fi * Fi_conj + (1 - self.learning_rate) * self.Bi

    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        channels, height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        # NOTE: xpos and ypos is the upper left corner of the bounding box.
        center_x = gt.xpos + 0.5 * gt.width
        center_y = gt.ypos + 0.5 * gt.height
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = linear_mapping(response)
        return response