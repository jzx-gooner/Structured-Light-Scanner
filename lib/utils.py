from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import open3d as o3d

@dataclass(frozen=True)
class Plane:
    origin: np.ndarray
    normal: np.ndarray
    R: Optional[np.ndarray] = None


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Rectangle:
    top_left: Point
    bottom_right: Point


@dataclass(frozen=True)
class ExtremePoints:
    wall: Rectangle
    desk: Rectangle


def draw_circles(image, points, text=False,color=(50, 50, 255)):
    """
    Given an image, draw colored circles on each point
    specified in `points`.
    If text is True, each corner is also tagged with an increasing number.
    """
    for i, point in enumerate(points):
        p = tuple([int(i) for i in point[0]])
        cv2.circle(image, p, 4, color, -1)
        if text:
            cv2.putText(
                image,
                text=str(i),
                org=p,
                fontFace=1,
                fontScale=1.5,
                color=color,
            )


def fit_plane(points):
    """
    Fit a plane through a bunch of 3D points.
    Return the plane as an (origin, normal) tuple.
    """
    mean = np.mean(points, axis=0)
    xx = 0
    xy = 0
    xz = 0
    yy = 0
    yz = 0
    zz = 0

    for point in points:
        diff = point - mean
        xx += diff[0] * diff[0]
        xy += diff[0] * diff[1]
        xz += diff[0] * diff[2]
        yy += diff[1] * diff[1]
        yz += diff[1] * diff[2]
        zz += diff[2] * diff[2]

    det_x = yy * zz - yz * yz
    det_y = xx * zz - xz * xz
    det_z = xx * yy - xy * xy
    det_max = max(det_x, det_y, det_z)

    if det_max == det_x:
        normal = np.array([det_x, xz * yz - xy * zz, xy * yz - xz * yy])
    elif det_max == det_y:
        normal = np.array([xz * yz - xy * zz, det_y, xy * xz - yz * xx])
    else:
        normal = np.array([xy * yz - xz * yy, xy * xz - yz * xx, det_z])

    normal = normal / np.linalg.norm(normal)

    return Plane(origin=np.array(mean), normal=normal)


def to_binary_image(img):
    """
    Given an image, return it as a B/W thresholded image.
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresholded = cv2.threshold(
        img_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    return img_thresholded


def line_plane_intersection(plane_origin, plane_normal, line_direction):
    """
    Given a point on a plane, the normal to the plane at that point and a ray in 3D,
    find the intersection point between the 3D ray and the 3D plane.
    """
    d = np.dot(plane_origin, plane_normal) / np.dot(line_direction, plane_normal)
    return line_direction * d


def load_intrinsics(debug=True):
    """
    Load the matrix of intrinsic parameters K and the five distortion parameters
    from the XML file saved after having performed camera calibration.
    """
    intrinsics = cv2.FileStorage("calibration/intrinsics.xml", cv2.FILE_STORAGE_READ)
    K = intrinsics.getNode("K").mat()
    dist = intrinsics.getNode("dist").mat()
    if debug:
        print(f"K:\n{K}")
        print(f"Distortion parameters:\n{dist}\n\n")
    return K, dist


def show_image(img, continuous=False):
    """
    Show an image. If continuous is True, the display
    is non-breaking and can be closed by hitting Q.
    """
    cv2.imshow("Image", img)
    if not continuous:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return cv2.waitKey(1) & 0xFF == ord("q")


def show_pointcloud(points,colors):
    """
    Given points in the 3D world, save the PLY file representing
    the point cloud. This function saves both the original file and
    a version to which an outlier removal process has been applied.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(points).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    vis.draw_geometries([pcd])
