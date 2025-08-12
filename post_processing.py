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























import os
import cv2
import math
from typing import List, Tuple, Optional
import numpy as np
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def detect_pl_stars_post_process(
    binary_mask: np.ndarray,
    hough_thresh: int = 20,
    min_len: int = 10,
    max_gap: int = 3,
    angle_tol: float = 1.0,
    cluster_eps: float = 10.0,
    cluster_min_pts: int = 3,
    visualize: bool = True,
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
        visualize: Whether to show visualization
    
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
    
    # Store intermediate results for visualization
    intermediate_results = {}
    
    # Step 1: Apply morphological closing operation to bridge small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    intermediate_results['closing'] = closed

    # Step 2: Skeletonize the mask to get single-pixel-width lines
    skel = skeletonize(closed > 0).astype(np.uint8) * 255
    intermediate_results['skeleton'] = skel

    # Step 3: Use probabilistic Hough Transform to detect lines
    lines_p = cv2.HoughLinesP(
        skel, 
        rho=1, 
        theta=np.pi/180, 
        threshold=hough_thresh,
        minLineLength=min_len, 
        maxLineGap=max_gap
    )
    
    # Store all detected lines for visualization
    all_lines = []
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            all_lines.append((x1, y1, x2, y2))
    intermediate_results['hough_lines'] = all_lines

    # Step 4: Filter lines based on angle tolerance
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
    
    intermediate_results['filtered_lines'] = kept

    # Step 5: Compute intersection points from filtered lines
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

    intermediate_results['intersections'] = intersection_points

    # Step 6: Cluster intersection points to find likely centers
    centers = []
    cluster_labels = []
    
    if intersection_points:
        points_array = np.array(intersection_points)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=cluster_eps,
            min_samples=cluster_min_pts
        ).fit(points_array)
        
        cluster_labels = clustering.labels_
        
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

    intermediate_results['cluster_labels'] = cluster_labels
    intermediate_results['centers'] = centers

    # Output detection results
    if centers:
        print(f"[INFO] Detected {len(centers)} PL Star centers")
        for i, (x, y) in enumerate(centers):
            print(f"    → Center {i+1}: ({x:.2f}, {y:.2f})")
    else:
        print("[INFO] No PL Star centers detected")
    
    # Visualize all steps if requested
    if visualize:
        visualize_all_steps(binary_mask, intermediate_results)
    
    return centers, kept, closed, skel


def visualize_all_steps(binary_mask: np.ndarray, results: dict) -> None:
    """
    Visualize all processing steps side by side.
    
    Args:
        binary_mask: Original binary mask
        results: Dictionary containing all intermediate results
    """
    # Create figure with 6 rows for all steps
    fig, axes = plt.subplots(6, 2, figsize=(12, 18))
    fig.suptitle('PL Star Detection Pipeline', fontsize=16, fontweight='bold')
    
    # Step 1: Morphological Closing
    axes[0, 0].imshow(binary_mask, cmap='gray')
    axes[0, 0].set_title('Original Binary Mask')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(results['closing'], cmap='gray')
    axes[0, 1].set_title('After Morphological Closing')
    axes[0, 1].axis('off')
    
    # Add text explanation
    fig.text(0.5, 0.845, 'Goal: Bridge small gaps and connect broken line segments', 
             ha='center', fontsize=10, color='blue')
    
    # Step 2: Skeletonization
    axes[1, 0].imshow(results['closing'], cmap='gray')
    axes[1, 0].set_title('After Closing')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(results['skeleton'], cmap='gray')
    axes[1, 1].set_title('After Skeletonization')
    axes[1, 1].axis('off')
    
    fig.text(0.5, 0.705, 'Goal: Reduce lines to single-pixel width for accurate line detection', 
             ha='center', fontsize=10, color='blue')
    
    # Step 3: Hough Transform (All detected lines)
    axes[2, 0].imshow(results['skeleton'], cmap='gray')
    axes[2, 0].set_title('Skeleton')
    axes[2, 0].axis('off')
    
    # Draw all detected lines
    temp_img = cv2.cvtColor(results['skeleton'], cv2.COLOR_GRAY2RGB)
    for x1, y1, x2, y2 in results['hough_lines']:
        cv2.line(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    axes[2, 1].imshow(temp_img)
    axes[2, 1].set_title(f'Hough Lines Detected ({len(results["hough_lines"])} lines)')
    axes[2, 1].axis('off')
    
    fig.text(0.5, 0.565, 'Goal: Detect all possible line segments in the skeleton', 
             ha='center', fontsize=10, color='blue')
    
    # Step 4: Filtered Lines (Angle-based filtering)
    axes[3, 0].imshow(temp_img)
    axes[3, 0].set_title('All Hough Lines')
    axes[3, 0].axis('off')
    
    # Draw filtered lines with different colors for different angles
    filtered_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    color_map = {0.0: (255, 0, 0), 60.0: (0, 255, 0), 120.0: (0, 0, 255)}
    for x1, y1, x2, y2, angle in results['filtered_lines']:
        color = color_map.get(angle, (255, 255, 0))
        cv2.line(filtered_img, (x1, y1), (x2, y2), color, 2)
    axes[3, 1].imshow(filtered_img)
    axes[3, 1].set_title(f'Filtered Lines ({len(results["filtered_lines"])} lines)\nRed:0°, Green:60°, Blue:120°')
    axes[3, 1].axis('off')
    
    fig.text(0.5, 0.425, 'Goal: Keep only lines aligned with PL star directions (0°, 60°, 120°)', 
             ha='center', fontsize=10, color='blue')
    
    # Step 5: Intersection Points
    axes[4, 0].imshow(filtered_img)
    axes[4, 0].set_title('Filtered Lines')
    axes[4, 0].axis('off')
    
    # Draw intersection points
    intersect_img = filtered_img.copy()
    for x, y in results['intersections']:
        cv2.circle(intersect_img, (int(x), int(y)), 3, (255, 255, 0), -1)
    axes[4, 1].imshow(intersect_img)
    axes[4, 1].set_title(f'Line Intersections ({len(results["intersections"])} points)')
    axes[4, 1].axis('off')
    
    fig.text(0.5, 0.285, 'Goal: Find all intersection points between filtered lines', 
             ha='center', fontsize=10, color='blue')
    
    # Step 6: Clustering and Final Centers
    axes[5, 0].imshow(intersect_img)
    axes[5, 0].set_title('Intersection Points')
    axes[5, 0].axis('off')
    
    # Draw clustered points and centers
    final_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    
    # Draw filtered lines
    for x1, y1, x2, y2, angle in results['filtered_lines']:
        color = color_map.get(angle, (255, 255, 0))
        cv2.line(final_img, (x1, y1), (x2, y2), color, 1)
    
    # Draw clustered intersection points
    if len(results['intersections']) > 0 and len(results['cluster_labels']) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for (x, y), label in zip(results['intersections'], results['cluster_labels']):
            if label >= 0:
                color = (colors[label % 10][:3] * 255).astype(int)
                cv2.circle(final_img, (int(x), int(y)), 2, color.tolist(), -1)
    
    # Draw final centers
    for x, y in results['centers']:
        cv2.circle(final_img, (int(x), int(y)), 8, (255, 0, 255), 2)
        cv2.circle(final_img, (int(x), int(y)), 2, (255, 0, 255), -1)
    
    axes[5, 1].imshow(final_img)
    axes[5, 1].set_title(f'Final Centers ({len(results["centers"])} stars)\nMagenta circles = star centers')
    axes[5, 1].axis('off')
    
    fig.text(0.5, 0.145, 'Goal: Cluster nearby intersections to find star centers', 
             ha='center', fontsize=10, color='blue')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def explain_parameters():
    """
    Explain the meaning and function of each parameter used in the detection.
    """
    explanations = """
    ================================================================================
    PARAMETER EXPLANATIONS FOR PL STAR DETECTION
    ================================================================================
    
    1. MORPHOLOGICAL CLOSING PARAMETERS:
    --------------------------------------------------------------------------------
    - kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
      → Shape: ELLIPSE - Provides smoother, more natural connections
      → Size: (3, 3) - Small kernel to bridge small gaps without over-dilating
      → Purpose: Connects broken line segments caused by noise or image artifacts
    
    2. HOUGH TRANSFORM PARAMETERS:
    --------------------------------------------------------------------------------
    - rho = 1: Distance resolution in pixels (1 pixel accuracy)
    - theta = np.pi/180: Angle resolution (1 degree accuracy)
    - threshold = 20: Minimum votes needed in accumulator to detect a line
      → Higher value: More strict, fewer false positives
      → Lower value: More sensitive, may detect noise as lines
    - minLineLength = 10: Minimum line segment length in pixels
      → Filters out very short segments that are likely noise
    - maxLineGap = 3: Maximum gap between line segments to connect them
      → Allows connecting slightly broken lines
    
    3. ANGLE FILTERING PARAMETERS:
    --------------------------------------------------------------------------------
    - canonical_angles = [0.0, 60.0, 120.0]: Expected PL star directions
      → PL stars have 6 rays at 60-degree intervals
      → We only need 3 angles since opposite rays share the same angle
    - angle_tol = 1.0: Tolerance in degrees for angle matching
      → Accounts for slight rotations or imperfect line detection
      → Too small: May miss valid lines
      → Too large: May include incorrect lines
    
    4. DBSCAN CLUSTERING PARAMETERS:
    --------------------------------------------------------------------------------
    - eps = 10.0: Maximum distance between points to be in same cluster
      → Controls how close intersection points must be to group together
      → Larger value: Merges distant points (may combine separate stars)
      → Smaller value: Creates more clusters (may split single stars)
    - min_samples = 3: Minimum points needed to form a cluster
      → A valid star needs at least 3 intersections (from 3 line directions)
      → Filters out spurious intersections from only 2 lines
    
    ================================================================================
    PROCESSING PIPELINE FLOW:
    ================================================================================
    
    Binary Mask → Closing → Skeleton → Hough Lines → Filter by Angle → 
    Find Intersections → Cluster Points → Identify Star Centers
    
    Each step refines the data to progressively identify the PL star patterns:
    1. Closing: Repairs broken patterns
    2. Skeleton: Simplifies to essential structure
    3. Hough: Detects linear segments
    4. Filter: Keeps only star-aligned segments
    5. Intersect: Finds potential star centers
    6. Cluster: Groups intersections into star centers
    ================================================================================
    """
    print(explanations)


# Example usage
if __name__ == "__main__":
    # First, explain all parameters
    explain_parameters()
    
    # Create a sample binary mask with multiple stars
    mask = np.zeros((300, 400), dtype=np.uint8)
    
    # Draw multiple star patterns with some imperfections
    stars = [
        (100, 100, 40),  # (center_x, center_y, length)
        (250, 150, 35),
        (150, 220, 30)
    ]
    
    for center_x, center_y, length in stars:
        # Draw 6 lines (0°, 60°, 120° directions and their opposite directions)
        angles = [0, 60, 120]
        for angle in angles:
            rad = math.radians(angle)
            x1 = int(center_x - length * math.cos(rad))
            y1 = int(center_y - length * math.sin(rad))
            x2 = int(center_x + length * math.cos(rad))
            y2 = int(center_y + length * math.sin(rad))
            
            # Add some imperfections (gaps in lines)
            if np.random.random() > 0.3:  # 70% chance to draw full line
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
            else:  # 30% chance to draw line with gap
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                gap = 5
                cv2.line(mask, (x1, y1), (mid_x - gap, mid_y - gap), 255, 2)
                cv2.line(mask, (mid_x + gap, mid_y + gap), (x2, y2), 255, 2)
    
    # Add some noise
    noise = np.random.random((300, 400)) < 0.001
    mask[noise] = 255
    
    # Detect PL stars with visualization
    centers, lines, closed, skeleton = detect_pl_stars_post_process(
        mask, 
        visualize=True,
        hough_thresh=15,
        min_len=8,
        max_gap=5,
        angle_tol=2.0,
        cluster_eps=15.0,
        cluster_min_pts=3
    )