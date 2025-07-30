#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from collections import deque
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image
from vision_msgs.msg import BoundingBox2D
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import open3d as o3d

class PlaneFittingNode(Node):
    def __init__(self):
        super().__init__('plane_fitting_node')

        # ===== 可調參數 (Adjustable Parameters) =====
        # RANSAC 平面擬合參數
        self.declare_parameter('distance_threshold', 0.01)    # 最大距離閥值 (公尺)
        self.declare_parameter('ransac_n', 3)                # RANSAC 每次抽樣點數
        self.declare_parameter('num_iterations', 1000)       # RANSAC 迭代次數
        # 中值濾波緩衝窗口 (幀數)
        self.declare_parameter('filter_window_size', 10)      # 平面參數中值濾波視窗長度
        # 離群值移除參數 (Statistical Outlier Removal)
        self.declare_parameter('outlier_nb_neighbors', 20)   # 用於統計濾波的鄰居數
        self.declare_parameter('outlier_std_ratio', 2.0)     # 標準差比率閥值

        # 取出參數值到屬性，方便後續使用
        self.distance_threshold   = self.get_parameter('distance_threshold').value
        self.ransac_n             = self.get_parameter('ransac_n').value
        self.num_iterations       = self.get_parameter('num_iterations').value
        self.filter_window_size   = self.get_parameter('filter_window_size').value
        self.outlier_nb_neighbors = self.get_parameter('outlier_nb_neighbors').value
        self.outlier_std_ratio    = self.get_parameter('outlier_std_ratio').value

        # ===== 資料訂閱 (Subscriptions) =====
        self.sub_roi   = self.create_subscription(BoundingBox2D, 'spd/plane_roi',        self.roi_callback, 10)
        self.sub_pc    = self.create_subscription(PointCloud2,    'tm_robot/pointcloud', self.pc_callback, 10)
        self.sub_color = self.create_subscription(Image,           'tm_robot/color_image', self.color_callback, 10)

        # ===== 資料發布 (Publishers) =====
        self.pub_center    = self.create_publisher(Point,      'spd/plane_center',  10)
        self.pub_normal    = self.create_publisher(Vector3,    'spd/plane_normal',  10)
        self.pub_marker    = self.create_publisher(Marker,     'spd/plane_marker',  10)
        self.pub_roi_image = self.create_publisher(Image,      'spd/color_image',   10)

        # ===== 內部狀態 (Internal State) =====
        self.bridge = CvBridge()
        self.latest_roi   = None
        self.latest_pc    = None
        self.latest_image = None
        # 用 deque 做中值濾波緩衝 (plane center & normal)
        self.center_buffer = deque(maxlen=self.filter_window_size)
        self.normal_buffer = deque(maxlen=self.filter_window_size)

    def roi_callback(self, msg: BoundingBox2D):
        """接收 ROI，並嘗試繪製 & 發佈"""
        self.latest_roi = msg
        self.draw_and_publish_roi()
        self.try_process()

    def pc_callback(self, msg: PointCloud2):
        """接收點雲，並嘗試處理"""
        self.latest_pc = msg
        self.try_process()

    def color_callback(self, msg: Image):
        """接收彩色影像，並嘗試繪製 ROI"""
        self.latest_image = msg
        self.draw_and_publish_roi()

    def try_process(self):
        """當 ROI 與點雲都已接收到，再執行擬合"""
        if self.latest_roi is None or self.latest_pc is None:
            return
        self.fit_plane(self.latest_pc, self.latest_roi)
        # 處理完畢後重置
        self.latest_roi = None
        self.latest_pc  = None

    def draw_and_publish_roi(self):
        """將 ROI 畫到影像上並發布"""
        if self.latest_roi is None or self.latest_image is None:
            return
        cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        h, w = cv_img.shape[:2]
        cx = int(self.latest_roi.center.position.x)
        cy = int(self.latest_roi.center.position.y)
        sx = int(self.latest_roi.size_x / 2)
        sy = int(self.latest_roi.size_y / 2)
        xmin = max(0, cx - sx)
        xmax = min(w, cx + sx)
        ymin = max(0, cy - sy)
        ymax = min(h, cy + sy)
        # 畫綠色方框
        cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # 發佈 ROI 圖片
        img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        img_msg.header = self.latest_image.header
        self.pub_roi_image.publish(img_msg)

    def fit_plane(self, pc_msg: PointCloud2, roi_msg: BoundingBox2D):
        """對 ROI 範圍內的點雲做離群值移除、RANSAC 擬合，並中值濾波結果"""
        # --- 1) 獲取點雲尺寸 ---
        if pc_msg.height > 1:
            H, W = pc_msg.height, pc_msg.width
        else:
            if self.latest_image is None:
                self.get_logger().warn('未取得影像，無法裁切 unorganized 點雲')
                return
            H, W = self.latest_image.height, self.latest_image.width

        # --- 2) 計算 ROI 像素邊界 ---
        cx = int(roi_msg.center.position.x)
        cy = int(roi_msg.center.position.y)
        sx = int(roi_msg.size_x / 2)
        sy = int(roi_msg.size_y / 2)
        xmin, xmax = max(0, cx - sx), min(W, cx + sx)
        ymin, ymax = max(0, cy - sy), min(H, cy + sy)
        self.get_logger().info(f'ROI: x[{xmin},{xmax}) y[{ymin},{ymax}), size=({W},{H})')

        # --- 3) 提取 ROI 點 ---
        region_pts = []
        for idx, p in enumerate(pc2.read_points(pc_msg, field_names=('x','y','z'), skip_nans=False)):
            row = idx // W
            if row < ymin:
                continue
            if row >= ymax:
                break
            col = idx % W
            if col < xmin or col >= xmax:
                continue
            if not all(np.isfinite(v) for v in p):
                continue
            region_pts.append(p)
        points = np.array(region_pts, dtype=np.float32)
        self.get_logger().info(f'裁切後點數: {points.shape[0]}')
        if points.shape[0] < 50:
            self.get_logger().warn('點數不足，無法擬合')
            return

        # --- 4) 離群值移除 ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_filtered, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio
        )
        inlier_pts = np.asarray(pcd_filtered.points, dtype=np.float32)
        self.get_logger().info(f'移除離群後點數: {inlier_pts.shape[0]}')
        if inlier_pts.shape[0] < 50:
            self.get_logger().warn('移除後點數不足，無法擬合')
            return

        # --- 5) RANSAC 平面擬合 ---
        model, inliers = pcd_filtered.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations
        )
        a, b, c, d = model
        normal = np.array([a, b, c], dtype=np.float32)
        normal /= np.linalg.norm(normal)
        center = np.mean(np.asarray(pcd_filtered.select_by_index(inliers).points), axis=0)
        # 確保法線朝向 Sensor
        if np.dot(normal, -center) < 0:
            normal = -normal

        # --- 6) 緩衝 & 中值濾波 ---
        self.center_buffer.append(center)
        self.normal_buffer.append(normal)
        if len(self.center_buffer) == self.filter_window_size:
            filtered_center = np.median(np.vstack(self.center_buffer), axis=0)
            filtered_normal = np.median(np.vstack(self.normal_buffer), axis=0)
            filtered_normal /= np.linalg.norm(filtered_normal)
        else:
            filtered_center, filtered_normal = center, normal

        # --- 7) 發佈結果 ---
        self.pub_center.publish(Point(x=float(filtered_center[0]),
                                     y=float(filtered_center[1]),
                                     z=float(filtered_center[2])))
        self.pub_normal.publish(Vector3(x=float(filtered_normal[0]),
                                       y=float(filtered_normal[1]),
                                       z=float(filtered_normal[2])))
        # Marker: 用 Arrow 顯示法線方向
        marker = Marker()
        marker.header = pc_msg.header
        marker.ns     = 'fitted_plane'
        marker.id     = 0
        marker.type   = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [
            Point(x=float(filtered_center[0]), y=float(filtered_center[1]), z=float(filtered_center[2])),  # 起點
            Point(x=float(filtered_center[0] + filtered_normal[0]*0.1),
                  y=float(filtered_center[1] + filtered_normal[1]*0.1),
                  z=float(filtered_center[2] + filtered_normal[2]*0.1))  # 終點
        ]
        marker.scale.x = 0.01  # 箭頭桿粗細
        marker.scale.y = 0.02  # 箭頭頭部寬度
        marker.scale.z = 0.0   # 不使用
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # 全不透明
        self.pub_marker.publish(marker)
        self.get_logger().info(f'Filtered normal: {filtered_normal}, center: {filtered_center}')


def main(args=None):
    rclpy.init(args=args)
    node = PlaneFittingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
