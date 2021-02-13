import math
import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from lib.utils import (
    ExtremePoints,
    Plane,
    Point,
    Rectangle,
    draw_circles,
    fit_plane,
    line_plane_intersection,
    show_image,
    to_binary_image,
)


class Scanner3D:
    def __init__(self,debug,K,K_inv,dist,filename):
        self.debug = debug
        self.K = K
        self.K_inv = K_inv
        self.dist = dist 
        self.filename = filename 
        self.inner_rectangle = np.array([[[0, 0]], [[23, 0]], [[23, 13]], [[0, 13]]])
        self.lower_red_obj = np.array([0, 0, 230]) #np.array([35, 25, 40])
        self.lower_red_planes = np.array([0, 0, 242]) #np.array([45, 30, 45])
        self.upper_red = np.array([179, 255, 255]) #np.array([100, 255, 255])
        self.dbscan = DBSCAN(eps=12, min_samples=20)

    def get_rectangles_mask(self, thresh):
        show_image(thresh)
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        mask = np.zeros(thresh.shape, np.uint8)
        # for i in contours:
        #     print(cv2.contourArea(i))
        # return
        good_contours = sorted(
            [cnt for cnt in contours if 60000 < cv2.contourArea(cnt) < 80000],
            key=cv2.contourArea,
        )
        setattr(self, "contour1", good_contours[0])
        setattr(
            self,
            "contour2",
            good_contours[1]
            if cv2.pointPolygonTest(
                good_contours[1], tuple(good_contours[0][0][0]), False
            )
            < 0
            else good_contours[2],
        )

        cv2.drawContours(mask, [self.contour1], 0, 255, -1)
        cv2.drawContours(mask, [self.contour2], 0, 255, -1)
        return mask

    def sort_corners(self, corners):
        center = np.sum(corners, axis=0) / 4
        sorted_corners = sorted(
            corners,
            key=lambda p: math.atan2(p[0][0] - center[0][0], p[0][1] - center[0][1]),
            reverse=True,
        )
        return np.roll(sorted_corners, 1, axis=0) #

    def get_desk_wall_corners(self,thresh):
        mask = self.get_rectangles_mask(thresh)
        show_image(mask)
        assert thresh.shape[:2] == mask.shape[:2]
        corners = cv2.goodFeaturesToTrack(
            thresh,
            maxCorners=8,
            qualityLevel=0.01,
            minDistance=50,
            mask=mask,
            blockSize=5,
        )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 20, 0.001)
        corners = cv2.cornerSubPix(
            thresh, corners, winSize=(7, 7), zeroZone=(-1, -1), criteria=criteria
        )
        y_middle = thresh.shape[0] / 2
        desk_corners = np.expand_dims(corners[corners[:, :, 1] > y_middle], axis=1)
        wall_corners = np.expand_dims(corners[corners[:, :, 1] <= y_middle], axis=1)
        sorted_desk_corners = self.sort_corners(desk_corners)
        sorted_wall_corners = self.sort_corners(wall_corners)
        return sorted_desk_corners, sorted_wall_corners

    def get_H_R_t(self, corners):
        H = cv2.findHomography(self.inner_rectangle, corners)[0]
        result = self.K_inv @ H
        result /= cv2.norm(result[:, 1])
        r0, r1, t = np.hsplit(result, 3)
        r2 = np.cross(r0.T, r1.T).T
        _, u, vt = cv2.SVDecomp(np.hstack([r0, r1, r2]))
        R = u @ vt
        return Plane(origin=t[:, 0], normal=R[:, 2], R=R)

    def get_extreme_points(
        self, wall_corners, desk_corners
    ):
        ymin_wall = int(np.min(wall_corners[:, :, 1]))
        ymax_wall = int(np.max(wall_corners[:, :, 1]))
        ymin_desk = int(np.min(desk_corners[:, :, 1]))
        ymax_desk = int(np.max(desk_corners[:, :, 1]))
        xmin = int(np.min(wall_corners[:, :, 0]))
        xmax = int(np.max(wall_corners[:, :, 0]))
        return ExtremePoints(
            wall=Rectangle(
                top_left=Point(xmin, ymin_wall), bottom_right=Point(xmax, ymax_wall)
            ),
            desk=Rectangle(
                top_left=Point(xmin, ymin_desk), bottom_right=Point(xmax, ymax_desk)
            ),
        )

    def get_laser_points_in_region(
        self, image, region, is_obj=False,
    ):
        top_left = region.top_left
        bottom_right = region.bottom_right
        region_image = image[top_left.y : bottom_right.y, top_left.x : bottom_right.x]
        #image_inv = cv2.cvtColor(~region_image, cv2.COLOR_BGR2HSV)
        image_inv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
        lower_red = self.lower_red_obj if is_obj else self.lower_red_planes
        red_mask = cv2.inRange(image_inv, lower_red, self.upper_red)
        laser_points = cv2.findNonZero(red_mask)
        if laser_points is None or (not is_obj and len(laser_points) < 30):
            return None
        return laser_points

    def offset_points(self, points, offset):
        points[:, :, 0] += offset.x
        points[:, :, 1] += offset.y
        return points

    def make_homogeneous(self, points):
        return np.hstack((points[:, 0], np.ones(points.shape[0]).reshape(-1, 1),))

    def remove_obj_outliers(self, points):
        dbscan_result = self.dbscan.fit(points[:, 0])
        mask = dbscan_result.labels_ != -1
        return np.expand_dims(points[:, 0][mask], axis=1)

    def get_colors(self, image, coordinates):
        x = coordinates.squeeze(1)
        return np.flip(image[x[:, 1], x[:, 0]].astype(np.float64) / 255.0, axis=1)

    def get_laser_points(
        self,
        original_image,
        image,
        extreme_points,
    ):
        height, width = image.shape[:2]
        ymin_wall = extreme_points.wall.top_left.y
        ymax_wall = extreme_points.wall.bottom_right.y
        ymin_desk = extreme_points.desk.top_left.y
        xmin = extreme_points.desk.top_left.x
        laser_desk = self.get_laser_points_in_region(
            image=image,
            region=Rectangle(
                top_left=Point(0, ymin_desk - ymin_wall),
                bottom_right=Point(width, height),
            ),
        )
        if laser_desk is not None:
            laser_wall = self.get_laser_points_in_region(
                image=image,
                region=Rectangle(
                    top_left=Point(0, 0),
                    bottom_right=Point(width, ymax_wall - ymin_wall),
                ),
            )
            if laser_wall is not None:
                laser_obj = self.get_laser_points_in_region(
                    image=image,
                    region=Rectangle(
                        top_left=Point(0, ymax_wall - ymin_wall),
                        bottom_right=Point(width, ymin_desk - ymin_wall),
                    ),
                    is_obj=True,
                )
                if laser_obj is not None:
                    laser_desk = self.offset_points(
                        points=laser_desk, offset=Point(xmin, ymin_desk)
                    )
                    laser_wall = self.offset_points(
                        points=laser_wall, offset=Point(xmin, ymin_wall)
                    )
                    laser_obj = self.remove_obj_outliers(laser_obj)
                    if laser_obj is not None:
                        laser_obj = self.offset_points(
                            points=laser_obj, offset=Point(xmin, ymax_wall)
                        )
                        obj_colors = self.get_colors(original_image, laser_obj)
                        return laser_wall, laser_desk, laser_obj, obj_colors
        return None, None, None, None

    def save_3d_render(
        self, points, colors
    ):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(points).astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))
        if self.debug:
            o3d.visualization.draw_geometries([pcd])
        if not self.debug:
            o3d.io.write_point_cloud(f"results/{self.filename[:-4]}.ply", pcd)
    
    def show_3d_render(self,points,colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(points).astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))
        o3d.visualization.draw_geometries([pcd])



    def read_frame(self, cap):
        frame_raw = cap.read()[1]
        if frame_raw is None:
            cv2.destroyAllWindows()
            return None
        return cv2.undistort(frame_raw, self.K, self.dist)

    def create_exiting_rays(
        self, points, is_obj= False
    ):

        if not is_obj and len(points) > 100:
            points = points[np.random.choice(points.shape[0], 100, replace=False,)]
        return [self.K_inv @ point for point in points]

    def compute_intersections(
        self, plane, directions
    ):
        return [
            line_plane_intersection(
                plane_origin=plane.origin,
                plane_normal=plane.normal,
                line_direction=direction,
            )
            for direction in directions
        ]

    def run(self):
        #0. for video
        cap = cv2.VideoCapture(f"videos/{self.filename}")
        #1. for real time camera
        # cap = cv2.VideoCapture(2)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        if not cap.isOpened():
            return
        first_frame = self.read_frame(cap)
        if first_frame is None:
            return
        #0.灰度->二值图像
        first_frame_thresh = to_binary_image(first_frame)
        #1.get two rectangles corners
        desk_corners, wall_corners = self.get_desk_wall_corners(first_frame_thresh)
        #2.get object roi
        extreme_points = self.get_extreme_points(wall_corners, desk_corners)
        #3.get two plane
        desk_plane = self.get_H_R_t(desk_corners)
        wall_plane = self.get_H_R_t(wall_corners)

        all_obj_points = []
        all_obj_colors = []

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        raw_length = 0
        while True:
            frame = self.read_frame(cap)
            if frame is None:
                break
            frame_copy = frame.copy()
            cv2.drawContours(frame_copy, [self.contour1], -1, (255, 255, 255), 2)
            cv2.drawContours(frame_copy, [self.contour2], -1, (255, 255, 255), 2)
            draw_circles(frame_copy, desk_corners, text=True)
            draw_circles(frame_copy, wall_corners, text=True)
            #jiequ xia chang de na kuai qu yu
            frame_interesting = frame[
                extreme_points.wall.top_left.y : extreme_points.desk.bottom_right.y, #rows
                extreme_points.wall.top_left.x : extreme_points.wall.bottom_right.x, #colums
            ]
            (laser_wall, laser_desk, laser_obj, obj_colors,) = self.get_laser_points(
                first_frame, frame_interesting, extreme_points
            )
            if laser_wall is not None:
     
                draw_circles(frame_copy, laser_wall,text=False,color = (255,0,0))
                draw_circles(frame_copy, laser_desk,text=False,color = (0,255,0))
                draw_circles(frame_copy, laser_obj)
    

                laser_wall = self.make_homogeneous(laser_wall)
                laser_obj = self.make_homogeneous(laser_obj)
                laser_desk = self.make_homogeneous(laser_desk)

                # Given a set of 2D points, get their real world 3D coordinates (direction).
                wall_directions = self.create_exiting_rays(laser_wall, is_obj=False)
                desk_directions = self.create_exiting_rays(laser_desk, is_obj=False)
                obj_directions = self.create_exiting_rays(laser_obj, is_obj=True)

                intersections_wall = self.compute_intersections(
                    wall_plane, wall_directions
                )
                intersections_desk = self.compute_intersections(
                    desk_plane, desk_directions
                )
                intersections_rects = np.array(intersections_wall + intersections_desk)
                #fit laser plane 
                laser_plane = fit_plane(intersections_rects)
                #
                intersections_objs = self.compute_intersections(
                    laser_plane, obj_directions
                )
                all_obj_points.extend(intersections_objs)
                all_obj_colors.extend(obj_colors)
                if(len(all_obj_points)>1.1*raw_length):
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_obj_points).astype(np.float64))
                    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_obj_colors))
                    vis.add_geometry(pcd)
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    raw_length = len(all_obj_points)
            if self.debug:
                if show_image(frame_copy, continuous=True):
                    break
            else:
                if show_image(frame, continuous=True):
                    break
        vis.destroy_window()
        all_obj_points.append(np.array([0, 0, 0]))
        all_obj_colors.append(np.array([255, 0, 0]))
        self.save_3d_render(all_obj_points, all_obj_colors)
        cap.release()
        cv2.destroyAllWindows()
