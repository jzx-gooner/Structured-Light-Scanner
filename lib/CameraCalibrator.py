import click
import cv2
import numpy as np
import math
from utils import draw_circles, show_image, to_binary_image
import glob

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960


class ChessboardDetector:
    def __init__(self,base_folder,n_images, debug):
        self.base_folder=base_folder
        self.n_images = n_images
        self.destination=np.array([[[0, 900]], [[0, 0]], [[700, 0]], [[700, 900]]])
        self.debug = debug
        self.width=700
        self.height=900

    def generate_chessboard_corners(self):
        """
        Generate the 3D objectPoints vector needed for `calibrateCamera`.
        This is generated programmatically as it reproduces the SVG chessboard.
        """
        corners = (
            [[1, j, 0] for j in range(1, 12, 2)]
            + [[i, j, 0] for i in range(3, 14, 2) for j in range(1, 14, 2)]
            + [[15, j, 0] for j in range(1, 12, 2)]
            + [[17, j, 0] for j in range(3, 10, 2)]
        )
        return np.expand_dims(np.array(corners), axis=1).astype(np.float32)

    def get_chessboard_mask(self, thresh):
        """
        Given a B/W thresholded image `thresh`,
        return a mask which contains the chessboard and removes basic noise (such as X-Y symbols).
        The "internal" rectangle containing the chessboard is always the third one (after having
        sorted by area in descending order).
        """
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(thresh.shape, np.uint8)
        areas = sorted(
            [i for i in range(len(contours))],
            key=lambda i: cv2.contourArea(contours[i]),
            reverse=True,
        )
        chessboard = contours[areas[2]]
        cv2.drawContours(mask, [chessboard], 0, 255, -1)
        location_xy = None
        for cnt in contours:
            if (
                5 < cv2.contourArea(cnt) < 1000
                and cv2.pointPolygonTest(chessboard, tuple(cnt[0][0]), True) > 1
            ):
                location_xy = cnt[0].astype(np.float32)
                return mask, chessboard, location_xy

    def find_closest_point(self, points, target):
        """
        Given `n` points and a target point,
        return the point which is closest to the target
        """
        index = 0
        minimum = points[0]
        min_dist = cv2.norm(target, points[0])
        for i, point in enumerate(points[1:]):
            new_dist = cv2.norm(target, point)
            if new_dist < min_dist:
                min_dist = new_dist
                minimum = point
                index = i + 1
        return minimum, index

    def load_and_threshold_image(self, filename: str):
        """
        Given a path to an image, load it, threshold it and return the two results.
        """
        img = cv2.imread(filename)
        img_thresholded = to_binary_image(img)
        return img, img_thresholded

    def find_four_corners(self, mask: np.ndarray) -> np.ndarray:
        """
        Given a mask quite similar to a rectangle, return the four corners.
        """
        chessboard_corners = cv2.goodFeaturesToTrack(
            mask,
            maxCorners=4,
            qualityLevel=0.05,
            minDistance=100,
            mask=None,
            blockSize=5,
            gradientSize=3,
        )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 20, 0.001)
        chessboard_corners = cv2.cornerSubPix(
            mask,
            chessboard_corners,
            winSize=(3, 3),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
        return chessboard_corners

    def get_sorted_corners(self, chessboard_corners, location_xy):
        """
        Given four corners and the location of the "XY" sign, sort the corners in clockwise order,
        with the first corner being the one closer to the XY sign.
        """
        corner_0_0, index_0_0 = self.find_closest_point(chessboard_corners, location_xy)
        center = np.sum(chessboard_corners, axis=0) / 4
        sorted_corners = sorted(
            chessboard_corners,
            key=lambda p: math.atan2(p[0][0] - center[0][0], p[0][1] - center[0][1]),
            reverse=True,
        )
        while any(sorted_corners[0].ravel() != corner_0_0.ravel()):
            sorted_corners = np.roll(sorted_corners, 1)
        return sorted_corners

    def create_inner_mask(self, shape):
        """
        Given the shape of the straightened chessboard, create the mask.
        This removes 30px per border, as the actual squares of the chessboard start ~50 pixels
        from the border by construction.
        """
        mask = np.full(shape, 255, np.uint8)
        s1 = shape[1]
        s0 = shape[0]
        cv2.rectangle(mask, (0, 0), (s1, 30), 0, -1)
        cv2.rectangle(mask, (0, 0), (30, s0), 0, -1)
        cv2.rectangle(
            mask, (0, s0 - 30), (s1, s0), 0, -1,
        )
        cv2.rectangle(
            mask, (s1 - 30, 0), (s1, s0), 0, -1,
        )
        return mask

    def sort_inner_corners(self, inner_corners):
        """
        Given a `warped` rectified image of the chessboard's inner corners,
        sort the corners starting from the XY symbol on the bottom left of the chessboard.
        """
        sorted_corners = []
        for i in np.arange(start=self.height, stop=0, step=-100):
            sorted_corners += sorted(
                inner_corners[
                    np.logical_and(
                        inner_corners[:, :, 1] < i, inner_corners[:, :, 1] > i - 100
                    )
                ],
                key=lambda p: p[0],
                reverse=False,
            )
        return np.array(sorted_corners)

    def find_inner_corners_warped(self, warped_thresholded: np.ndarray):
        """
        Given a thresholded `warped` image obtained via homography, get the inner corners of the chessboard.
        """
        contours = cv2.findContours(
            warped_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        mask = self.create_inner_mask(warped_thresholded.shape)

        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)

        # smooth the edges a tad
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(warped_thresholded, cv2.MORPH_OPEN, kernel)

        inner_corners = cv2.goodFeaturesToTrack(
            opening,
            maxCorners=60,
            qualityLevel=0.01,
            minDistance=85,
            mask=mask,
            blockSize=3,
            gradientSize=9,
        )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 20, 0.001)
        inner_corners = cv2.cornerSubPix( #亚像素角点检测
            opening, inner_corners, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria
        )

        if self.debug and self.img_index % 5 == 0:
            opening_copy = opening.copy()
            for corner in inner_corners:
                cv2.circle(
                    img=opening_copy,
                    center=tuple(corner[0]),
                    radius=8,
                    color=(130, 255, 155),
                    thickness=-1,
                )
            show_image(opening_copy)

        sorted_corners = self.sort_inner_corners(inner_corners)
        return np.expand_dims(sorted_corners, axis=1).astype(np.float32)

    def run(self):
        result = []
        img_list = sorted(glob.glob(self.base_folder+"/*.jpg"))
        setattr(self, "img_index", 0) #增加新属性
        for self.img_index in range(self.n_images):
            print(img_list[self.img_index])
            img, img_thresholded = self.load_and_threshold_image(img_list[self.img_index])
            mask, chessboard, location_xy = self.get_chessboard_mask(img_thresholded)
    
            if self.debug and self.img_index % 1 == 0:
                img_copy = img.copy()
                cv2.drawContours(img_copy, [chessboard], -1, (236, 255, 122), 5)

            chessboard_corners = self.find_four_corners(mask)
            sorted_corners = self.get_sorted_corners(chessboard_corners, location_xy)

            if self.debug and self.img_index % 1 == 0:
                draw_circles(img_copy, sorted_corners)
                show_image(img_copy)
 
            H = cv2.findHomography(np.array(sorted_corners), self.destination)[0] #计算多个二维点对之间的最优单映射变换矩阵 H
            H_inv = np.linalg.inv(H) #矩阵求逆
            warped = cv2.warpPerspective(img, H, (self.width, self.height))#透视变换
            warped_thresh = to_binary_image(warped)  #灰度图
            inner_corners = self.find_inner_corners_warped(warped_thresh)#求角点
            inner_corners_in_original_image = []
            for i, corner in enumerate(inner_corners):
                corner_inv = H_inv @ np.vstack([corner[0][0], corner[0][1], [1]]) #叉乘 按垂直方向（行顺序）堆叠数组构成一个新的数组
                corner_inv = corner_inv[:2] / corner_inv[2]
                inner_corners_in_original_image.append([corner_inv])

            if self.debug and self.img_index % 1== 0:
                draw_circles(img_copy, inner_corners_in_original_image, text=True)
                show_image(img_copy)
            print(np.array(result).shape) 
            print(np.array([inner_corners_in_original_image]).shape)
            result = result + [inner_corners_in_original_image]
        return np.array(result).astype(np.float32)

class CameraCalibrator:
    def __init__(self,debug):
        self.debug = debug
        self.base_folder="calibration/images/"
        self.n_images=81

    def show_undistorted_images(self, K, dist):
        """
        This method shows three undistorted images to visually verify the quality of
        `K` and `dist`.
        """
        # for img_index in range(3):
        #     img = cv2.imread(self.base_folder + f"img_000{img_index:02d}.png")
        #     img_undistorted = cv2.undistort(img, K, dist) #矫正回去
        #     cv2.imshow(f"Undistorted_img_000{img_index:02d}", img_undistorted)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()

    def compute_intrinsics(self):
        """
        Compute the matrix of intrinsic parameters `K` and the 5 distortion parameters `dist`.
        """
        c = ChessboardDetector(
            base_folder=self.base_folder, n_images=self.n_images, debug=self.debug
        )

        image_points = c.run()
        image_points = np.reshape(
            image_points, (image_points.shape[0], image_points.shape[1], 1, 2)
        )
        object_points = np.array(
            [c.generate_chessboard_corners() for _ in range(image_points.shape[0])]
        )

        print("Calibrating camera...")

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, (IMAGE_WIDTH, IMAGE_HEIGHT), None, None
        )
        print(f"\n\nRMS: {ret}")
        print(f"\n\nK: {K}")
        print(f"Distortion parameters:\n{dist}")
        print(f"Images used for calibration: {image_points.shape[0]} out of 50")

        Kfile = cv2.FileStorage("calibration/intrinsics.xml", cv2.FILE_STORAGE_WRITE)
        Kfile.write("RMS", ret)
        Kfile.write("K", K)
        Kfile.write("dist", dist)

        if self.debug:
            self.show_undistorted_images(K, dist)



@click.command()
@click.option(
    "--debug", default=False, is_flag=True, help="Use this flag to show debug images"
)
def cli(debug):
    cc = CameraCalibrator(debug=debug)
    cc.compute_intrinsics()


if __name__ == "__main__":
    cli()
