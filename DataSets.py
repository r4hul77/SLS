import json

import cv2 as cv
import numpy as np
import pandas as pd



class WestMorelandDataSet:

    def __init__(self, videoPath, csvFile):

        self.video_capture = cv.VideoCapture(videoPath)
        self.data_frame    = pd.read_csv(csvFile)
        self.frame_idx     = 0
        self.frame_height  = self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.frame_width   = self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH)
        self.calib_props = None
        self.new_opt_mtx = None
        self.roi         = None
        self.end_frame = None
        self.start_frame = 0
        if(not self.video_capture.isOpened()):
            raise ValueError
        # TODO Write a check for if we are using the correct file

    @property
    def width(self):
        return self.frame_width

    @property
    def height(self):
        return self.frame_height

    def set_end_frame(self, end_frame):
        self.end_frame = end_frame

    def seek_frame(self, frame_no):

        if(len(self) < frame_no):
            raise ValueError
        else:
            self.video_capture.set(cv.CAP_PROP_POS_FRAMES, frame_no)
            self.frame_idx = frame_no
            self.start_frame = frame_no

    def seek_percent(self, percent):

        if(percent >= 1.0 ):
            raise ValueError

        else:

            frame_no = int(percent*len(self))
            self.seek_frame(frame_no)

    def get_acceleration(self, frame_idx, dt = 0.1):

        if frame_idx + 4 >= len(self.data_frame):
            return self.get_acceleration(frame_idx - 4)

        vel_now = self.get_velocity(frame_idx)

        vel_next = self.get_velocity(frame_idx + 4)

        return (vel_next - vel_now) / dt


    def get_heading(self, frame_idx):

        return self.data_frame['Amplitude - Heading'][frame_idx]

    def get_velocity(self, frame_idx):

        return 5/18*self.data_frame['Amplitude - Vel'][frame_idx]

    def get_long(self, frame_idx):

        return self.data_frame['Amplitude - Long'][frame_idx]

    def get_lat(self, frame_idx):

        return self.data_frame['Amplitude - Lat'][frame_idx]

    def preprocess_frame(self, frame):
        ret = np.vsplit(frame, 4)
        if(self.calib_props):
            ret = list(map(lambda x: self.undistort(x), ret))
        return np.vsplit(frame, 4)

    def set_calibartion(self, cam_props):
        self.calib_props = cam_props


    def load_calib_props_from_json(self, json_file):
        with open(json_file, "r"):
            self.calib_props = json.load(json_file)

        for key in self.calib_props.keys():
            self.calib_props[key] = np.asarray(self.calib_props[key])

    def undistort(self, img):
        if not self.new_opt_mtx:
            self.calculate_new_opt_mtx(img)
        dst = cv.undistort(img, self.calib_props["mtx"], self.calib_props["dist"], None, self.new_opt_mtx)

        dst = dst[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]]

        return dst

    def calculate_new_opt_mtx(self, img):
        h, w = img.shape[:2]
        self.new_opt_mtx, self.roi = cv.getOptimalNewCameraMatrix(self.calib_props["mtx"], self.calib_props["dist"], (w, h), 1, (w, h))

    def __len__(self):

        return int(self.video_capture.get(cv.CAP_PROP_FRAME_COUNT))

    def get_total_frames(self):
        return self.end_frame - self.start_frame

    def __iter__(self):

        return self


    def __next__(self):
        ret, frame = self.video_capture.read()
        if(not ret) :
            raise StopIteration

        accel  = self.get_acceleration(self.frame_idx)
        vel    = self.get_velocity(self.frame_idx)
        lat    = self.get_lat(self.frame_idx)
        long   = self.get_long(self.frame_idx)
        frames = self.preprocess_frame(frame)
        heading = self.get_heading(self.frame_idx)
        self.frame_idx += 1

        if(self.end_frame):
            if(self.frame_idx > self.end_frame):
                raise StopIteration

        return self.frame_idx, *frames, lat, long, vel, accel, heading


class GoodFieldDataSet(WestMorelandDataSet):

    def __init__(self, folder):
        super().__init__(folder + "/Video.avi", folder + "/GpsData.csv" )


    def get_acceleration(self, frame_idx, dt=0.2):
        if frame_idx + 8 >= len(self.data_frame):
            return self.get_acceleration(frame_idx - 8)

        vel_now = self.get_velocity(frame_idx)

        vel_next = self.get_velocity(frame_idx + 8)

        return (vel_next - vel_now) / dt


def writeText(frame, lat, long, vel, accel, frame_idx):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.2
    lines = 'Lat {}\nLong {}\nVel {}\nAccel {}\nFrame Idx {}\n '.format(lat, long, vel, accel, frame_idx)
    org = [0, 120]
    for i, line in enumerate(lines.split("\n")):
        frame = cv.putText(frame, line, org, font, font_scale, (255, 0, 0), 1, cv.LINE_8)
        org[1] += 10
    return frame

if __name__ == "__main__":

    folder_path = "/home/harsha/Desktop/SLS-CNH/Data/exp3/2-35"
    dataset = GoodFieldDataSet(folder=folder_path)

    for img_0, img_1, img_2, img_3, lat, long, vel, accel, heading in dataset:
        print(img_0)
        cv.imshow("Img 0 ", writeText(img_0, lat, long, vel, accel, dataset.frame_idx))
        cv.waitKey(100)