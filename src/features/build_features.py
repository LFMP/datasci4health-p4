from copy import deepcopy

import numpy as np
from glrlm import GLRLM
from scipy import stats
from skimage import util
from skimage.exposure import equalize_adapthist
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import MinMaxScaler


def get_lbp_with_mask_feature(image, mask, R):
  """
  Get the LBP feature of the given image.
  """
  P = R * 8
  lbp_mask = local_binary_pattern(image * mask, P, R, method='default')

  return lbp_mask


def get_equalized_hist_image(image):
  """
  Get the equalized histogram feature of the given image.
  """
  image = equalize_adapthist(image, clip_limit=0.03)
  return image


def median_mad_normalization(image):
  """
  Get the median and MAD of the given image.
  """
  median = np.median(image)
  mad = np.median(np.abs(image - median))
  norm = (image - median) / mad
  return norm


def get_normalized_images(image):
  """
  Get the normalized image of the given image.
  """
  scaler = MinMaxScaler([0, 1])
  scaler.fit(deepcopy(image))
  imgnorm_maxmin = scaler.transform(deepcopy(image))
  imgnorm_maxmin = util.img_as_ubyte(imgnorm_maxmin)
  imgnorm_maxmin = imgnorm_maxmin // 32

  imgnorm_zs = stats.zscore(deepcopy(image))
  imgnorm_zs = util.img_as_ubyte(imgnorm_zs)
  imgnorm_zs = imgnorm_zs // 32

  imgnorm_medianmad = median_mad_normalization(deepcopy(image))
  imgnorm_medianmad = util.img_as_ubyte(imgnorm_medianmad / 255)
  imgnorm_medianmad = imgnorm_medianmad // 32

  return imgnorm_maxmin, imgnorm_zs, imgnorm_medianmad


def GLCM_feature(image):
  distances_list = [1, 4, 8]
  angles = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi, 3 * np.pi / 2]
  glcm = graycomatrix(image=image,
                      distances=distances_list,
                      angles=angles,
                      levels=256,
                      symmetric=True,
                      normed=True)
  return graycoprops(glcm, prop='contrast')


def GLRLM_feature(image, distance=2):
  app = GLRLM()

  glrlm = app.get_features(image, distance)
  features = {
      "Short Run Emphasis (SRE)": glrlm.Features[0],
      "Long Run Emphasis (LRE)": glrlm.Features[1],
      "Grey Level Uniformity (GLU)": glrlm.Features[2],
      "Run Length Uniformity (RLU)": glrlm.Features[3],
      "Run Percentage (RPC)": glrlm.Features[4],
  }
  return features


def LBP_image(image, r, method='default'):
  n_points = 8 * r
  lbp = local_binary_pattern(image, n_points, r, method=method)
  h_img, _ = np.histogram(lbp.ravel(),
                          bins=np.arange(0, n_points + 3),
                          range=(0, n_points + 2))
  h_img = h_img.astype(float)
  h_img = h_img / (h_img.sum(dtype=float) + 1e-7)
  return lbp, h_img


def get_histogram(image):
  h_img, _ = np.histogram(image.ravel(), bins=np.arange(0, 256 + 1))
  h_img = h_img.astype(float)
  h_img = h_img / (h_img.sum(dtype=float) + 1e-7)
  return h_img
