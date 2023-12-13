import cv2
import numpy as np
from turbojpeg import TurboJPEG


def test_create_context(ctx):
    pass


def test_bind_data(ctx):
    ctx.bind_data(env_map_path="tests/data/test_env_map.hdr")


def test_create_program(ctx):
    ctx.create_program()


def test_render(ctx):
    ctx.render(1)


def get_mssim(src1, src2):
    gaussian_kernel_size = (11, 11)
    gaussian_sigma = 1.5
    C1 = 6.5025  # (K_1 * L) ** 2; K_1 = 0.01, L = 255
    C2 = 58.5225  # (K_2 * L) ** 2; K_2 = 0.03, L = 255

    I1 = src1.astype("float32")
    I2 = src2.astype("float32")
    I1_2 = I1**2
    I2_2 = I2**2
    I1_I2 = I1 * I2

    mu1 = cv2.GaussianBlur(I1, gaussian_kernel_size, gaussian_sigma)
    mu2 = cv2.GaussianBlur(I2, gaussian_kernel_size, gaussian_sigma)
    mu1_2 = mu1**2
    mu2_2 = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, gaussian_kernel_size, gaussian_sigma)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, gaussian_kernel_size, gaussian_sigma)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, gaussian_kernel_size, gaussian_sigma)
    sigma12 -= mu1_mu2
    t1 = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    t2 = (mu1_2 + mu2_2 + C1) * (sigma1_2 + sigma2_2 + C2)
    ssim = t1 / t2
    mssim = ssim.mean()

    return mssim


def test_get_binary(ctx):
    reference = cv2.imread("tests/data/reference.jpg", cv2.IMREAD_COLOR)
    rendered = TurboJPEG().decode(np.frombuffer(ctx.get_binary(), dtype=np.uint8))
    mssim = get_mssim(reference, rendered)

    print("\nSSIM: " + str(mssim) + "\n")

    assert mssim > 0.98
