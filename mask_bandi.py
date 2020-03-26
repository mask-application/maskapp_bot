import argparse
import glob
import os
import time

import cv2
import dlib
import face_alignment
import numpy as np
from skimage.transform import warp, PolynomialTransform, PiecewiseAffineTransform

LEFT_EAR_TOP = 0
LEFT_EAR_UPPER_MIDDLE = 1
LEFT_EAR_LOWER_MIDDLE = 2
LEFT_EAR_BOTTOM = 3
RIGHT_EAR_TOP = 16
RIGHT_EAR_UPPER_MIDDLE = 15
RIGHT_EAR_LOWER_MIDDLE = 14
RIGHT_EAR_BOTTOM = 13
NOSE_BOTTOM = 33
NOSE_MIDDLE = 28
NOSE_TOP = 27
LIP = 57
LEFT_CHIN = 5
LEFT_CHIN_1 = 6
LEFT_CHIN_2 = 7
CHIN = 8
RIGHT_CHIN_2 = 9
RIGHT_CHIN_1 = 10
RIGHT_CHIN = 11
LEFT_EYEBROW = 19
RIGH_EYEBROW = 24

MASK_BANDI_PATH = "./mask_bandi.png"
MASK_BANDI_COORDINATES = [[351, 505], [335, 717], [1149, 505], [1165, 717],
                          [750, 960], [430, 880], [1070, 880],
                          [560, 310], [940, 310],
                          [750, 375], [750, 550],
                          ]
MASK_BANDI_CORRESPONDING = [LEFT_EAR_TOP, LEFT_EAR_BOTTOM, RIGHT_EAR_TOP, RIGHT_EAR_BOTTOM,
                            CHIN, LEFT_CHIN, RIGHT_CHIN,
                            LEFT_EYEBROW, RIGH_EYEBROW,
                            NOSE_MIDDLE, NOSE_BOTTOM,
                            ]
MASK_BANDI_COORDINATES_AFFINE = MASK_BANDI_COORDINATES + [
    [500, 900], [600, 930], [900, 930], [1000, 900]
]
MASK_BANDI_CORRESPONDING_AFFINE = MASK_BANDI_CORRESPONDING + [
    LEFT_CHIN_1, LEFT_CHIN_2, RIGHT_CHIN_2, RIGHT_CHIN_1
]

DECORATION_PATH = "./circle.png"

DLIB_PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"


class MaskBandi:
    def __init__(self, mask_images, mask_src_coordinates, mask_dst_coordinates, decoration_img=None,
                 face_detection_method="fa", transformation="polynomial", debug=False):
        self.mask_images = mask_images
        self.mask_src_coordinates = mask_src_coordinates
        self.mask_dst_coordinates = mask_dst_coordinates
        self.decoration_img = decoration_img
        self.face_detection_method = face_detection_method
        self.transformation = transformation
        self.debug = debug

        if self.face_detection_method == "fa":
            self.face_alignment = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                               device='cpu', flip_input=False)
        elif self.face_detection_method == "dlib":
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)

    def get_face_landmarks(self, img):
        if self.face_detection_method == "fa":
            faces = self.face_alignment.get_landmarks_from_image(img)
            if faces is None:
                return []
            return faces
        elif self.face_detection_method == "dlib":
            faces = []
            detected_faces = self.detector(img, 1)
            for face in detected_faces:
                face_shape = self.predictor(img, face)
                faces.append(np.asarray([(face_shape.part(i).x, face_shape.part(i).y)
                                         for i in range(face_shape.num_parts)]))
            return faces
        return []

    def add_mask(self, img):
        faces = self.get_face_landmarks(img)

        for face in faces:
            for mask_img, mask_src_cor, mask_dst_cor in \
                    zip(self.mask_images, self.mask_src_coordinates, self.mask_dst_coordinates):
                src_pts = np.array(mask_src_cor, dtype=np.float32)
                dst_pts = np.array([face[i] for i in mask_dst_cor], dtype=np.float32)

                if self.transformation == "polynomial":
                    transform = PolynomialTransform()
                    transform.estimate(dst_pts, src_pts, order=2)
                elif self.transformation == "affine":
                    transform = PiecewiseAffineTransform()
                    transform.estimate(dst_pts, src_pts)

                if self.debug:
                    for point in mask_src_cor:
                        mask_img = cv2.circle(mask_img, tuple(point), 10, (0, 255, 0, 255), -1)

                mask_aligned = warp(mask_img, transform, output_shape=(img.shape[0], img.shape[1])) * 255

                if not self.debug:
                    mask_aligned[:int(face[:, 1].min() - 5), :, :] = 0
                    mask_aligned[int(face[:, 1].max() + 5):, :, :] = 0
                    mask_aligned[:, :int(face[:, 0].min() - 5), :] = 0
                    mask_aligned[:, int(face[:, 0].max() + 5):, :] = 0

                alpha = mask_aligned[:, :, [3]] / 255.
                img = (img * (1 - alpha)) + mask_aligned[:, :, :3] * alpha

            img = img.astype(np.uint8)

            if self.debug:
                for landmark in face:
                    img = cv2.circle(img, (landmark[0], landmark[1]), 1, (0, 0, 255), -1)

                for mask_dst_cor in self.mask_dst_coordinates:
                    for i in mask_dst_cor:
                        img = cv2.circle(img, (face[i][0], face[i][1]), 2, (255, 0, 0), -1)

        return img

    @staticmethod
    def crop_center(img):
        y, x, _ = img.shape
        crop_size = min(x, y)
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[starty:starty + crop_size, startx:startx + crop_size, :]

    def add_decoration(self, img):
        img = self.crop_center(img)

        y, x, _ = img.shape
        y_d, x_d, _ = self.decoration_img.shape

        src_pts = np.array([[0, 0], [x_d - 1, 0], [x_d - 1, y_d - 1], [0, y_d - 1]], dtype=np.float32)
        dst_pts = np.array([[0, 0], [x - 1, 0], [x - 1, y - 1], [0, y - 1]], dtype=np.float32)

        transformation_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        mask_aligned = cv2.warpPerspective(self.decoration_img, transformation_matrix, (x, y))

        alpha = mask_aligned[:, :, [3]] / 255.
        img = (img * (1 - alpha)) + mask_aligned[:, :, :3] * alpha
        img = img.astype(np.uint8)

        return img


def main(source, input_dir=None, output_dir=None, decorate=False, method=1, debug=False):
    if method == 1:
        mask_bandi = MaskBandi(mask_images=[cv2.imread(MASK_BANDI_PATH, cv2.IMREAD_UNCHANGED)],
                               mask_src_coordinates=[MASK_BANDI_COORDINATES],
                               mask_dst_coordinates=[MASK_BANDI_CORRESPONDING],
                               decoration_img=cv2.imread(DECORATION_PATH, cv2.IMREAD_UNCHANGED),
                               face_detection_method="fa", transformation="polynomial", debug=debug)
    elif method == 2:
        mask_bandi = MaskBandi(mask_images=[cv2.imread(MASK_BANDI_PATH, cv2.IMREAD_UNCHANGED)],
                               mask_src_coordinates=[MASK_BANDI_COORDINATES],
                               mask_dst_coordinates=[MASK_BANDI_CORRESPONDING],
                               decoration_img=cv2.imread(DECORATION_PATH, cv2.IMREAD_UNCHANGED),
                               face_detection_method="dlib", transformation="polynomial", debug=debug)
    elif method == 3:
        mask_bandi = MaskBandi(mask_images=[cv2.imread(MASK_BANDI_PATH, cv2.IMREAD_UNCHANGED)],
                               mask_src_coordinates=[MASK_BANDI_COORDINATES_AFFINE],
                               mask_dst_coordinates=[MASK_BANDI_CORRESPONDING_AFFINE],
                               decoration_img=cv2.imread(DECORATION_PATH, cv2.IMREAD_UNCHANGED),
                               face_detection_method="fa", transformation="affine", debug=debug)
    elif method == 4:
        mask_bandi = MaskBandi(mask_images=[cv2.imread(MASK_BANDI_PATH, cv2.IMREAD_UNCHANGED)],
                               mask_src_coordinates=[MASK_BANDI_COORDINATES_AFFINE],
                               mask_dst_coordinates=[MASK_BANDI_CORRESPONDING_AFFINE],
                               decoration_img=cv2.imread(DECORATION_PATH, cv2.IMREAD_UNCHANGED),
                               face_detection_method="dlib", transformation="affine", debug=debug)
    if source == "webcam":
        webcam = cv2.VideoCapture(0)

        while True:
            time.sleep(1)
            rval, frame = webcam.read()

            result = mask_bandi.add_mask(frame)
            if decorate:
                result = mask_bandi.add_decoration(result)

            key = cv2.waitKey(20)
            if key in [27, ord('Q'), ord('q')]:  # exit on ESC/Q/q
                break

            cv2.imshow('Mask Bandi', result)

        webcam.release()
        cv2.destroyAllWindows()
    elif source == "file":
        assert input_dir is not None
        assert output_dir is not None
        img = cv2.imread(input_dir)
        result = mask_bandi.add_mask(img)
        if decorate:
            result = mask_bandi.add_decoration(result)
        cv2.imwrite(output_dir, result)
    elif source == "dir":
        assert input_dir is not None
        assert output_dir is not None
        for ext in ('*.gif', '*.png', '*.jpg'):
            for filename in glob.glob(os.path.join(input_dir, ext)):
                img = cv2.imread(filename)
                result = mask_bandi.add_mask(img)
                if decorate:
                    result = mask_bandi.add_decoration(result)
                cv2.imwrite(os.path.join(output_dir, filename.split("/")[-1]), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Mask Bandi')
    parser.add_argument('--source', type=str, default="webcam", choices=["webcam", "file", "dir"])
    parser.add_argument('-i', '--input', type=str, required=False)
    parser.add_argument('-o', '--output', type=str, required=False)
    parser.add_argument('--method', default=1, type=int, required=False)
    parser.add_argument('--decorate', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    face_alignment_dir = os.path.abspath(".")
    os.environ["FACEALIGNMENT_USERDIR"] = face_alignment_dir

    main(source=args.source, input_dir=args.input, output_dir=args.output, decorate=args.decorate,
         method=args.method, debug=args.debug)
