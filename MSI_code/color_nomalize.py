import cv2 as cv
import os
from PIL import Image
from torchvision import transforms
import numpy as np


def normal_Macenko(img, HERef, maxCRef, Io=255, alpha=1, beta=0.15):
    img = img.astype(np.float64)
    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float64) + 1) / Io)

    OD = OD.astype(np.float64)
    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # add a dimension according to the MATLAB code
    # by cross the H & E

    HE2 = np.cross(HE[:, 0], HE[:, 1])
    HE2 = HE2 / (HE2 ** 2).sum() ** 0.5
    HE = np.column_stack((HE, HE2))

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations

    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99), np.percentile(C[2, :], 99)])

    tmp = np.divide(maxC, maxCRef)

    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))

    Inorm[Inorm > 255] = 254

    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    return Inorm


def get_args():
    parser = argparse.ArgumentParser(
        description='The script to color normalize.')
    parser.add_argument(
        '-i', '--input_dir', default="/data/gbw/512_orginal_pic",
        help='Train_dir Path')
    parser.add_argument('-o', '--output_dir', default="/data/gbw/224_orginal_pic",
                        help='Test_dir Path')

    return parser.parse_args()

if __name__ == ('__main__'):
    args = get_args()
    maxCRef = np.load("/data/gbw/MCO/data/ref/maxCRef.npy")
    HERef = np.load("/data/gbw/MCO/data/ref/HERef.npy")

    datalist1 = os.listdir(args.input_dir)
    for j in datalist1:
        save_path = args.output_dir+"/" + j
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        datalist2 = os.listdir(args.input_dir+"/" + str(j))
        for i in datalist2:
            img1 = Image.open(args.input_dir + "/" + str(j) + "/" + str(i))
            np_patch = np.array(img1)
            np_patch = cv.cvtColor(np_patch, cv.COLOR_RGBA2RGB)
            np_patch = cv.resize(np_patch, (224, 224))
            try:
                img = normal_Macenko(np_patch, HERef, maxCRef, Io=255, alpha=1, beta=0.15)
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                cv.imwrite(save_path + "/" + str(i), img)
            except:
                print(i)