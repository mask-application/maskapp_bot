import argparse

import cv2
import dlib
import numpy as np
import glob
import os

from skimage.transform import PiecewiseAffineTransform, warp, PolynomialTransform

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"

LEFT_EAR_TOP = 0
LEFT_EAR_UPPER_MIDDLE = 1
LEFT_EAR_LOWER_MIDDLE = 2
LEFT_EAR_BOTTOM = 3
RIGHT_EAR_TOP = 16
RIGHT_EAR_UPPER_MIDDLE = 15
RIGHT_EAR_LOWER_MIDDLE = 14
RIGHT_EAR_BOTTOM = 13
NOSE_BOTTOM = 33
NOSE_MIDDLE = 29
NOSE_TOP = 27
LIP = 57
CHIN = 8
CHIN_LEFT = 6
CHIN_RIGHT = 10

MASK_BANDI_PATH = "./mask_bandi.png"
MASK_BANDI_COORDINATES = [[345, 475], [355, 770], [1155, 475], [1145, 770],
                          [750, 290], [750, 870]]
MASK_BANDI_CORRESPONDING = [LEFT_EAR_TOP, LEFT_EAR_BOTTOM, RIGHT_EAR_TOP, RIGHT_EAR_BOTTOM,
                            NOSE_TOP, CHIN]


DECORATION_PATH = "./circle.png"


class MaskBandi:
    def __init__(self, mask_images, mask_src_coordinates, mask_dst_coordinates,
                 decoration_img=None, debug=False):
        self.mask_images = mask_images
        self.mask_src_coordinates = mask_src_coordinates
        self.mask_dst_coordinates = mask_dst_coordinates
        self.decoration_img = decoration_img
        self.debug = debug

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

    def add_mask(self, img):
        faces = self.detector(img, 1)
        for face in faces:
            face_shape = self.predictor(img, face)

            for mask_img, mask_src_cor, mask_dst_cor in \
                    zip(self.mask_images, self.mask_src_coordinates, self.mask_dst_coordinates):
                src_pts = np.array(mask_src_cor, dtype=np.float32)
                dst_pts = np.array([(face_shape.part(i).x, face_shape.part(i).y) for i in mask_dst_cor],
                                   dtype=np.float32)

                transform = PolynomialTransform()
                transform.estimate(dst_pts, src_pts)

                mask_aligned = warp(mask_img, transform, output_shape=(img.shape[0], img.shape[1])) * 255
                mask_aligned[:min([face_shape.part(i).y for i in range(face_shape.num_parts)]), :, :] = 0
                mask_aligned[max([face_shape.part(i).y for i in range(face_shape.num_parts)]):, :, :] = 0
                mask_aligned[:, :min([face_shape.part(i).x for i in range(face_shape.num_parts)]), :] = 0
                mask_aligned[:, max([face_shape.part(i).x for i in range(face_shape.num_parts)]):, :] = 0

                # transformation_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                # mask_aligned = cv2.warpPerspective(mask_img, transformation_matrix, (img.shape[1], img.shape[0]))

                alpha = mask_aligned[:, :, [3]] / 255.
                img = (img * (1 - alpha)) + mask_aligned[:, :, :3] * alpha

            if self.debug:
                for i in range(face_shape.num_parts):
                    # print(f"Part({i}): {shape.part(i)}")
                    img = cv2.circle(img, (face_shape.part(i).x, face_shape.part(i).y), 1, (0, 0, 255), -1)

                for i in self.mask_dst_coordinates[0]:
                    # print(f"Part({i}): {shape.part(i)}")
                    img = cv2.circle(img, (face_shape.part(i).x, face_shape.part(i).y), 2, (255, 0, 0), -1)

            img = img.astype(np.uint8)

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
    if method == 3:
        mask_bandi = MaskBandi(mask_images=[cv2.imread(MASK_BANDI_PATH, cv2.IMREAD_UNCHANGED)],
                               mask_src_coordinates=[MASK_BANDI_COORDINATES],
                               mask_dst_coordinates=[MASK_BANDI_CORRESPONDING],
                               decoration_img=cv2.imread(DECORATION_PATH, cv2.IMREAD_UNCHANGED),
                               debug=debug)
    if source == "webcam":
        webcam = cv2.VideoCapture(0)

        while True:
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
    parser.add_argument('--method', default=3, type=int, required=False)
    parser.add_argument('--decorate', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(source=args.source, input_dir=args.input, output_dir=args.output, decorate=args.decorate,
         method=args.method, debug=args.debug)
