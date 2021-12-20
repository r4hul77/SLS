import cv2 as cv
import numpy as np
import json



def load_cam_props(cam_file):
    with open(cam_file, "r") as f:
        cam_props = json.load(f)

    for key in cam_props.keys():
        cam_props[key] = np.asarray(cam_props[key])
    return  cam_props

class CameraCalibration:

    def __init__(self, calibartion_json, h, w):
        cam_props = load_cam_props(calibartion_json)
        self.newOptMatrix, self.roi = cv.getOptimalNewCameraMatrix(cam_props["mtx"], cam_props["dist"], (w, h), 1, (w, h))
        self.cam_props = cam_props

    def undistort(self, img):
        return cv.undistort(img, self.cam_props["mtx"], self.cam_props["dist"], None, self.newOptMatrix)


if __name__ == "__main__":
    cam_props = load_cam_props("/home/harsha/Desktop/SLS-CNH/Data/exp3/Camera_20_calib.json")

    img = cv.imread("/home/harsha/Desktop/SLS-CNH/Data/exp3/2-34/0/163.jpeg")

    cv.imshow("Distored Img", img)

    h, w, c = img.shape

    newOptMatrix, roi = cv.getOptimalNewCameraMatrix(cam_props["mtx"], cam_props["dist"], (w, h), 1, (w, h))
    dst = cv.undistort(img, cam_props["mtx"], cam_props["dist"], None, newOptMatrix)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    print(img.shape)
    print(dst.shape)
    cv.imshow("unDistored Img", dst)
    cv.waitKey(0)