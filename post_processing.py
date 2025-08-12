# import os, cv2, yaml, math
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from skimage.morphology import skeletonize
# from sklearn.cluster import DBSCAN
# from scipy.io import loadmat

# def detect_pl_stars_post_process(
#     binary_mask,
#     hough_thresh = 20,
#     min_len = 10,
#     max_gap = 3,
#     angle_tol = 1,
#     cluster_eps = 10,
#     cluster_min_pts = 3,
# ):
#     # Apply morphological closing operation to bridge small gaps
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     closed = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

#     # Skeletonize the mask to get single-pixel-width lines
#     skel = skeletonize(closed > 0).astype(np.uint8) * 255   

#     # Use probabilistic Hough Transform to detect lines
#     lines_p = cv2.HoughLinesP(
#         skel, rho=1, theta=np.pi/180, threshold=hough_thresh,
#         minLineLength=min_len, maxLineGap=max_gap
#     )

#     # Filter lines based on angle toerlance
#     kept = []
#     if lines_p is not None:
#         canonical = np.array([0.0, 60.0, 120.0])
#         for x1, y1, x2, y2 in lines_p[:, 0, :]:
#             ang = (math.degrees(math.atan2(y2 - y1, x2 - x1)) + 360) % 180
#             diffs = np.abs(ang - canonical)
#             diffs = np.minimum(diffs, 180 - diffs)
#             idx = int(np.argmin(diffs))
#             if diffs[idx] < angle_tol:
#                 kept.append((x1, y1, x2, y2, float(canonical[idx])))
    
#     # Compute intersection points from filtered lines
#     pts, dir_pairs = [], []
#     line_eq = []
#     for x1, y1, x2, y2, ang0 in kept:
#         a, b = y2 - y1, x1 - x2
#         c = (x2 * y1 - x1 * y2)
#         line_eq.append((a, b, c, ang0))

#     for i in range(len(line_eq)):
#         a1, b1, c1, d1 = line_eq[i]
#         for j in range(i+1, len(line_eq)):
#             a2, b2, c2, d2 = line_eq[j]
#             D = a1 * b2 - a2 * b1
#             if abs(D) < 1e-6:
#                 continue
#             x = (b1 * c2 - b2 * c1) / D
#             y = (a2 * c1 - a1 * c2) / D
#             pts.append([x, y])
#             dir_pairs.append((d1, d2))

#     # Cluster intersection points to find likely center
#     centers = []
#     if pts:
#         pts_arr = np.array(pts)
#         clustering = DBSCAN(
#             eps = cluster_eps,
#             min_samples = cluster_min_pts
#         ).fit(pts_arr)
#         for lbl in set(clustering.labels_):
#             if lbl < 0:
#                 continue
#             mask_lbl = clustering.labels_ == lbl
#             dirs = set()
#             for k, pair in enumerate(dir_pairs):
#                 if mask_lbl[k]:
#                     dirs.update(pair)
#             if len(dirs) >= 3:
#                 centeroid = pts_arr[mask_lbl].mean(axis=0)
#                 centers.append((centeroid[0], centeroid[1]))

#     if centers:
#         print(f"[INFO] Detected {len(centers)} PL Star centers")
#         for i, (x, y) in enumerate(centers):
#             print(f"    → Center {i+1}: ({x:.2f}, {y:.2f})")
#     else:
#         print("[INFO] No PL Star centers detected")
#     return centers, kept, closed, skel

                

import os
import cv2
import math
from typing import List, Tuple, Optional
import numpy as np
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN


def detect_pl_stars_post_process(
    binary_mask: np.ndarray,
    hough_thresh: int = 20,
    min_len: int = 10,
    max_gap: int = 3,
    angle_tol: float = 1.0,
    cluster_eps: float = 10.0,
    cluster_min_pts: int = 3,
) -> Tuple[List[Tuple[float, float]], List[Tuple], np.ndarray, np.ndarray]:
    """
    检测二值掩码中的PL星形图案中心。
    
    该函数通过以下步骤检测星形图案：
    1. 形态学闭运算填补小间隙
    2. 骨架化得到单像素宽度的线条
    3. 使用概率霍夫变换检测直线
    4. 基于角度容差过滤线条（0°、60°、120°方向）
    5. 计算线条交点
    6. 使用DBSCAN聚类找到星形中心
    
    Args:
        binary_mask: 输入的二值掩码（numpy数组）
        hough_thresh: 霍夫变换的累加器阈值
        min_len: 最小线段长度
        max_gap: 线段间最大允许间隙
        angle_tol: 角度容差（度）
        cluster_eps: DBSCAN聚类的邻域半径
        cluster_min_pts: DBSCAN聚类的最小点数
    
    Returns:
        centers: 检测到的星形中心坐标列表
        kept: 保留的线段列表
        closed: 闭运算后的图像
        skel: 骨架化后的图像
    """
    # 验证输入
    if binary_mask is None or binary_mask.size == 0:
        raise ValueError("输入的二值掩码不能为空")
    
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # 应用形态学闭运算以桥接小间隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 骨架化掩码以获得单像素宽度的线
    skel = skeletonize(closed > 0).astype(np.uint8) * 255   

    # 使用概率霍夫变换检测线条
    lines_p = cv2.HoughLinesP(
        skel, 
        rho=1, 
        theta=np.pi/180, 
        threshold=hough_thresh,
        minLineLength=min_len, 
        maxLineGap=max_gap
    )

    # 基于角度容差过滤线条
    kept = []
    canonical_angles = np.array([0.0, 60.0, 120.0])  # PL星形的典型角度
    
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            
            # 计算线段角度（0-180度范围）
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle = (angle + 360) % 180
            
            # 找到最接近的典型角度
            angle_diffs = np.abs(angle - canonical_angles)
            angle_diffs = np.minimum(angle_diffs, 180 - angle_diffs)
            min_idx = np.argmin(angle_diffs)
            
            # 如果角度差在容差范围内，保留该线段
            if angle_diffs[min_idx] < angle_tol:
                kept.append((x1, y1, x2, y2, float(canonical_angles[min_idx])))
    
    # 计算过滤后线条的交点
    intersection_points = []
    direction_pairs = []
    line_equations = []
    
    # 计算每条线的方程系数 (ax + by + c = 0)
    for x1, y1, x2, y2, angle in kept:
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        line_equations.append((a, b, c, angle))

    # 计算所有线对的交点
    for i in range(len(line_equations)):
        a1, b1, c1, angle1 = line_equations[i]
        for j in range(i + 1, len(line_equations)):
            a2, b2, c2, angle2 = line_equations[j]
            
            # 计算行列式（检查线是否平行）
            determinant = a1 * b2 - a2 * b1
            
            # 如果线不平行，计算交点
            if abs(determinant) > 1e-6:
                x = (b1 * c2 - b2 * c1) / determinant
                y = (a2 * c1 - a1 * c2) / determinant
                
                # 检查交点是否在图像边界内
                h, w = binary_mask.shape
                if 0 <= x < w and 0 <= y < h:
                    intersection_points.append([x, y])
                    direction_pairs.append((angle1, angle2))

    # 使用DBSCAN聚类交点以找到可能的中心
    centers = []
    
    if intersection_points:
        points_array = np.array(intersection_points)
        
        # 执行DBSCAN聚类
        clustering = DBSCAN(
            eps=cluster_eps,
            min_samples=cluster_min_pts
        ).fit(points_array)
        
        # 处理每个聚类
        unique_labels = set(clustering.labels_)
        for label in unique_labels:
            # 跳过噪声点（标签为-1）
            if label < 0:
                continue
            
            # 获取属于当前聚类的点
            cluster_mask = clustering.labels_ == label
            cluster_points = points_array[cluster_mask]
            
            # 收集该聚类中涉及的所有方向
            unique_directions = set()
            for idx, (dir1, dir2) in enumerate(direction_pairs):
                if cluster_mask[idx]:
                    unique_directions.add(dir1)
                    unique_directions.add(dir2)
            
            # 如果涉及至少3个不同方向，认为是有效的星形中心
            if len(unique_directions) >= 3:
                centroid = cluster_points.mean(axis=0)
                centers.append((float(centroid[0]), float(centroid[1])))

    # 输出检测结果
    if centers:
        print(f"[INFO] 检测到 {len(centers)} 个PL星形中心")
        for i, (x, y) in enumerate(centers):
            print(f"    → 中心 {i+1}: ({x:.2f}, {y:.2f})")
    else:
        print("[INFO] 未检测到PL星形中心")
    
    return centers, kept, closed, skel


def visualize_results(
    original_image: np.ndarray,
    centers: List[Tuple[float, float]],
    kept_lines: List[Tuple],
    skeleton: Optional[np.ndarray] = None
) -> None:
    """
    可视化检测结果。
    
    Args:
        original_image: 原始图像
        centers: 检测到的中心点
        kept_lines: 保留的线段
        skeleton: 骨架化图像（可选）
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, axes = plt.subplots(1, 2 if skeleton is None else 3, figsize=(15, 5))
    
    # 显示原始图像
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示检测结果
    ax_idx = 1
    if skeleton is not None:
        axes[ax_idx].imshow(skeleton, cmap='gray')
        axes[ax_idx].set_title('骨架化图像')
        axes[ax_idx].axis('off')
        ax_idx += 1
    
    # 显示线段和中心点
    axes[ax_idx].imshow(original_image, cmap='gray')
    
    # 绘制检测到的线段
    for x1, y1, x2, y2, angle in kept_lines:
        axes[ax_idx].plot([x1, x2], [y1, y2], 'g-', linewidth=2, alpha=0.7)
    
    # 绘制检测到的中心点
    for x, y in centers:
        circle = mpatches.Circle((x, y), radius=5, color='red', fill=True)
        axes[ax_idx].add_patch(circle)
        axes[ax_idx].text(x + 7, y, f'({x:.1f}, {y:.1f})', 
                          color='red', fontsize=8)
    
    axes[ax_idx].set_title(f'检测结果 ({len(centers)} 个中心)')
    axes[ax_idx].axis('off')
    
    plt.tight_layout()
    plt.show()


# 示例用法
if __name__ == "__main__":
    # 创建一个示例二值掩码（实际使用时替换为真实数据）
    # 这里创建一个包含星形图案的简单示例
    mask = np.zeros((200, 200), dtype=np.uint8)
    
    # 绘制一个简单的星形图案
    center_x, center_y = 100, 100
    length = 40
    
    # 绘制6条线（0°, 60°, 120°方向及其相反方向）
    angles = [0, 60, 120]
    for angle in angles:
        rad = math.radians(angle)
        x1 = int(center_x - length * math.cos(rad))
        y1 = int(center_y - length * math.sin(rad))
        x2 = int(center_x + length * math.cos(rad))
        y2 = int(center_y + length * math.sin(rad))
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    
    # 检测PL星形
    centers, lines, closed, skeleton = detect_pl_stars_post_process(mask)
    
    # 可视化结果（如果需要）
    # visualize_results(mask, centers, lines, skeleton)



import os
import cv2
import math
from typing import List, Tuple, Optional
import numpy as np
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN


def detect_pl_stars_post_process(
    binary_mask: np.ndarray,
    hough_thresh: int = 20,
    min_len: int = 10,
    max_gap: int = 3,
    angle_tol: float = 1.0,
    cluster_eps: float = 10.0,
    cluster_min_pts: int = 3,
) -> Tuple[List[Tuple[float, float]], List[Tuple], np.ndarray, np.ndarray]:
    """
    Detect PL star pattern centers in a binary mask.
    
    This function detects star patterns through the following steps:
    1. Apply morphological closing to bridge small gaps
    2. Skeletonize to get single-pixel-width lines
    3. Use probabilistic Hough transform to detect lines
    4. Filter lines based on angle tolerance (0°, 60°, 120° directions)
    5. Compute line intersections
    6. Use DBSCAN clustering to find star centers
    
    Args:
        binary_mask: Input binary mask (numpy array)
        hough_thresh: Accumulator threshold for Hough transform
        min_len: Minimum line segment length
        max_gap: Maximum allowed gap between line segments
        angle_tol: Angle tolerance in degrees
        cluster_eps: DBSCAN clustering neighborhood radius
        cluster_min_pts: DBSCAN minimum number of points
    
    Returns:
        centers: List of detected star center coordinates
        kept: List of retained line segments
        closed: Image after closing operation
        skel: Skeletonized image
    """
    # Validate input
    if binary_mask is None or binary_mask.size == 0:
        raise ValueError("Input binary mask cannot be empty")
    
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # Apply morphological closing operation to bridge small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Skeletonize the mask to get single-pixel-width lines
    skel = skeletonize(closed > 0).astype(np.uint8) * 255   

    # Use probabilistic Hough Transform to detect lines
    lines_p = cv2.HoughLinesP(
        skel, 
        rho=1, 
        theta=np.pi/180, 
        threshold=hough_thresh,
        minLineLength=min_len, 
        maxLineGap=max_gap
    )

    # Filter lines based on angle tolerance
    kept = []
    canonical_angles = np.array([0.0, 60.0, 120.0])  # Canonical angles for PL star
    
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle (0-180 degree range)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle = (angle + 360) % 180
            
            # Find closest canonical angle
            angle_diffs = np.abs(angle - canonical_angles)
            angle_diffs = np.minimum(angle_diffs, 180 - angle_diffs)
            min_idx = np.argmin(angle_diffs)
            
            # Keep line if angle difference is within tolerance
            if angle_diffs[min_idx] < angle_tol:
                kept.append((x1, y1, x2, y2, float(canonical_angles[min_idx])))
    
    # Compute intersection points from filtered lines
    intersection_points = []
    direction_pairs = []
    line_equations = []
    
    # Calculate line equation coefficients (ax + by + c = 0)
    for x1, y1, x2, y2, angle in kept:
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        line_equations.append((a, b, c, angle))

    # Calculate intersections for all line pairs
    for i in range(len(line_equations)):
        a1, b1, c1, angle1 = line_equations[i]
        for j in range(i + 1, len(line_equations)):
            a2, b2, c2, angle2 = line_equations[j]
            
            # Calculate determinant (check if lines are parallel)
            determinant = a1 * b2 - a2 * b1
            
            # If lines are not parallel, calculate intersection
            if abs(determinant) > 1e-6:
                x = (b1 * c2 - b2 * c1) / determinant
                y = (a2 * c1 - a1 * c2) / determinant
                
                # Check if intersection is within image bounds
                h, w = binary_mask.shape
                if 0 <= x < w and 0 <= y < h:
                    intersection_points.append([x, y])
                    direction_pairs.append((angle1, angle2))

    # Cluster intersection points to find likely centers
    centers = []
    
    if intersection_points:
        points_array = np.array(intersection_points)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=cluster_eps,
            min_samples=cluster_min_pts
        ).fit(points_array)
        
        # Process each cluster
        unique_labels = set(clustering.labels_)
        for label in unique_labels:
            # Skip noise points (label -1)
            if label < 0:
                continue
            
            # Get points belonging to current cluster
            cluster_mask = clustering.labels_ == label
            cluster_points = points_array[cluster_mask]
            
            # Collect all unique directions involved in this cluster
            unique_directions = set()
            for idx, (dir1, dir2) in enumerate(direction_pairs):
                if cluster_mask[idx]:
                    unique_directions.add(dir1)
                    unique_directions.add(dir2)
            
            # If at least 3 different directions are involved, consider it a valid star center
            if len(unique_directions) >= 3:
                centroid = cluster_points.mean(axis=0)
                centers.append((float(centroid[0]), float(centroid[1])))

    # Output detection results
    if centers:
        print(f"[INFO] Detected {len(centers)} PL Star centers")
        for i, (x, y) in enumerate(centers):
            print(f"    → Center {i+1}: ({x:.2f}, {y:.2f})")
    else:
        print("[INFO] No PL Star centers detected")
    
    return centers, kept, closed, skel


def visualize_results(
    original_image: np.ndarray,
    centers: List[Tuple[float, float]],
    kept_lines: List[Tuple],
    skeleton: Optional[np.ndarray] = None
) -> None:
    """
    Visualize detection results.
    
    Args:
        original_image: Original image
        centers: Detected center points
        kept_lines: Retained line segments
        skeleton: Skeletonized image (optional)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, axes = plt.subplots(1, 2 if skeleton is None else 3, figsize=(15, 5))
    
    # Display original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display detection results
    ax_idx = 1
    if skeleton is not None:
        axes[ax_idx].imshow(skeleton, cmap='gray')
        axes[ax_idx].set_title('Skeletonized Image')
        axes[ax_idx].axis('off')
        ax_idx += 1
    
    # Display lines and center points
    axes[ax_idx].imshow(original_image, cmap='gray')
    
    # Draw detected line segments
    for x1, y1, x2, y2, angle in kept_lines:
        axes[ax_idx].plot([x1, x2], [y1, y2], 'g-', linewidth=2, alpha=0.7)
    
    # Draw detected center points
    for x, y in centers:
        circle = mpatches.Circle((x, y), radius=5, color='red', fill=True)
        axes[ax_idx].add_patch(circle)
        axes[ax_idx].text(x + 7, y, f'({x:.1f}, {y:.1f})', 
                          color='red', fontsize=8)
    
    axes[ax_idx].set_title(f'Detection Results ({len(centers)} centers)')
    axes[ax_idx].axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create a sample binary mask (replace with real data in actual use)
    # Here we create a simple example containing a star pattern
    mask = np.zeros((200, 200), dtype=np.uint8)
    
    # Draw a simple star pattern
    center_x, center_y = 100, 100
    length = 40
    
    # Draw 6 lines (0°, 60°, 120° directions and their opposite directions)
    angles = [0, 60, 120]
    for angle in angles:
        rad = math.radians(angle)
        x1 = int(center_x - length * math.cos(rad))
        y1 = int(center_y - length * math.sin(rad))
        x2 = int(center_x + length * math.cos(rad))
        y2 = int(center_y + length * math.sin(rad))
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    
    # Detect PL stars
    centers, lines, closed, skeleton = detect_pl_stars_post_process(mask)
    
    # Visualize results (if needed)
    # visualize_results(mask, centers, lines, skeleton)