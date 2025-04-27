import argparse
import numpy as np
import torch
import os
import mouette as M
from common.models import load_model
from common.utils import get_device
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import ast

def create_plane_from_parameters(location=[0,0,0], rotation=[0,0,0], size=1.0, resolution=100):
    """
    根据位置和旋转角度创建一个3D平面
    
    参数:
    location (list/tuple): 平面中心位置 [x, y, z]
    rotation (list/tuple): 平面旋转角度（欧拉角，弧度制）[rx, ry, rz]
    size (float): 平面边长
    resolution (int): 每边的采样分辨率
    
    返回:
    points (numpy.ndarray): 采样点坐标，形状为 (resolution*resolution, 3)
    points_grid (numpy.ndarray): 采样点的网格坐标，形状为 (resolution, resolution, 3)
    """
    # 将位置和旋转转换为numpy数组
    location = np.array(location, dtype=np.float32)
    rotation = np.array(rotation, dtype=np.float32)
    
    # 创建旋转矩阵（欧拉角转换为旋转矩阵）
    # 按ZYX顺序应用欧拉角（与Blender的XYZ顺序相反）
    rx, ry, rz = rotation
    
    # 围绕X轴的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # 围绕Y轴的旋转矩阵
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # 围绕Z轴的旋转矩阵
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # 计算组合旋转矩阵（按Z-Y-X顺序）
    R = Rz @ Ry @ Rx
    
    # 创建平面网格（中心在原点）
    uv = np.linspace(-size/2, size/2, resolution)
    U, V = np.meshgrid(uv, uv)
    
    points_grid = np.zeros((resolution, resolution, 3))
    for i in range(resolution):
        for j in range(resolution):
            # 初始平面在XY平面上
            local_point = np.array([U[i, j], V[i, j], 0.0])
            # 应用旋转
            rotated_point = R @ local_point
            # 应用平移
            world_point = rotated_point + location
            points_grid[i, j] = world_point
    
    # 重塑为点列表
    points = points_grid.reshape(-1, 3)
    
    return points, points_grid

# 保留旧的函数以保持兼容性
def create_plane_from_normal(normal, size=1.0, resolution=100):
    """
    根据法线方向创建一个3D平面
    
    参数:
    normal (list/tuple): 平面法线方向 [nx, ny, nz]
    size (float): 平面边长
    resolution (int): 每边的采样分辨率
    
    返回:
    points (numpy.ndarray): 采样点坐标，形状为 (resolution*resolution, 3)
    points_grid (numpy.ndarray): 采样点的网格坐标，形状为 (resolution, resolution, 3)
    """
    # 确保法线是单位向量
    normal = np.array(normal, dtype=np.float32)
    normal = normal / np.linalg.norm(normal)
    
    # 创建平面的基向量（正交于法线）
    if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
        # 法线沿Z轴时，使用X和Y轴作为基向量
        basis1 = np.array([1, 0, 0])
        basis2 = np.array([0, 1, 0])
    else:
        # 计算第一个基向量（任意与法线垂直的向量）
        basis1 = np.array([1, 1, -(normal[0] + normal[1]) / normal[2]]) if normal[2] != 0 else np.array([-(normal[1] + normal[2]) / normal[0], 1, 1])
        basis1 = basis1 / np.linalg.norm(basis1)  # 归一化
        
        # 计算第二个基向量（叉乘确保三个向量互相垂直）
        basis2 = np.cross(normal, basis1)
        basis2 = basis2 / np.linalg.norm(basis2)  # 归一化
    
    # 创建平面网格（中心在原点）
    uv = np.linspace(-size/2, size/2, resolution)
    U, V = np.meshgrid(uv, uv)
    
    points_grid = np.zeros((resolution, resolution, 3))
    for i in range(resolution):
        for j in range(resolution):
            # 在平面上生成点（中心在原点）
            u = U[i, j]
            v = V[i, j]
            points_grid[i, j] = u * basis1 + v * basis2
    
    # 重塑为点列表
    points = points_grid.reshape(-1, 3)
    
    return points, points_grid

def evaluate_sdf_on_plane(model, points, device, batch_size=1024):
    """
    在平面上的点上评估SDF模型
    
    参数:
    model: SDF模型
    points (numpy.ndarray): 平面上的采样点
    device: 计算设备
    batch_size (int): 批处理大小
    
    返回:
    values (numpy.ndarray): 每个采样点的SDF值
    """
    model.eval()
    values = np.zeros(points.shape[0])
    
    with torch.no_grad():
        for i in range(0, points.shape[0], batch_size):
            batch_points = torch.tensor(points[i:i+batch_size], dtype=torch.float32).to(device)
            batch_values = model(batch_points)
            values[i:i+batch_size] = batch_values.cpu().numpy().flatten()
    
    return values

def save_slice_plane_obj(points_grid, filename):
    """
    将切片平面保存为OBJ文件
    
    参数:
    points_grid (numpy.ndarray): 网格化的点坐标
    filename (str): 输出文件名
    """
    resolution = points_grid.shape[0]
    
    with open(filename, 'w') as f:
        # 写入顶点
        for i in range(resolution):
            for j in range(resolution):
                point = points_grid[i, j]
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        
        # 写入UV坐标 (归一化到0-1范围)
        for i in range(resolution):
            v = i / (resolution - 1)
            for j in range(resolution):
                u = j / (resolution - 1)
                f.write(f"vt {u} {v}\n")
        
        # 写入面 (三角形)
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # 当前格子的四个顶点索引 (注意OBJ文件中顶点索引从1开始)
                v0 = i * resolution + j + 1
                v1 = i * resolution + j + 2
                v2 = (i + 1) * resolution + j + 1
                v3 = (i + 1) * resolution + j + 2
                
                # 两个三角形组成一个格子
                f.write(f"f {v0}/{v0} {v2}/{v2} {v1}/{v1}\n")
                f.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")
    
    return resolution * resolution  # 返回顶点总数，方便后续存储SDF值

def normalize_values(values, vmin=None, vmax=None):
    """
    归一化SDF值到[0, 1]范围
    
    参数:
    values (numpy.ndarray): SDF值
    vmin, vmax (float): 手动指定的值范围，如果为None则从数据中计算
    
    返回:
    normalized (numpy.ndarray): 归一化后的值
    """
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    
    normalized = (values - vmin) / (vmax - vmin)
    return normalized

def visualize_sdf_slice(values_grid, output_path, vmin=None, vmax=None, cmap="RdBu_r", dpi=300):
    """
    将SDF值可视化为图像并保存为PNG
    
    参数:
    values_grid (numpy.ndarray): SDF值网格
    output_path (str): 输出图像路径
    vmin, vmax (float): 色彩映射范围
    cmap (str): matplotlib色彩映射名称
    dpi (int): 图像分辨率
    """
    plt.figure(figsize=(10, 8))
    
    # 设置vmin和vmax，确保有足够的范围防止colorbar错误
    if vmin is None:
        vmin = values_grid.min()
    if vmax is None:
        vmax = values_grid.max()
    
    # 如果vmin和vmax太接近，扩大范围
    if np.isclose(vmin, vmax) or abs(vmax - vmin) < 1e-6:
        mean = (vmin + vmax) / 2
        vmin = mean - 0.1
        vmax = mean + 0.1
        print(f"Warning: SDF values have very small range. Expanding to [{vmin}, {vmax}] for visualization.")
    
    # 创建主热图
    im = plt.imshow(values_grid, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
    
    # 计算适当的等值线级别
    try:
        # 尝试自动计算等值线级别
        min_val = values_grid.min()
        max_val = values_grid.max()
        step = (max_val - min_val) / 15  # 尝试15个等值线
        
        # 如果范围太小，使用固定间隔
        if step < 1e-6:
            levels = np.linspace(min_val - 0.1, max_val + 0.1, 15)
        else:
            levels = np.arange(np.floor(min_val / step) * step, 
                              np.ceil(max_val / step) * step, step)
        
        # 添加等值线
        if len(levels) > 1:
            contour = plt.contour(values_grid, levels=levels, colors='black', alpha=0.5, linewidths=0.5)
        
        # 添加零等值线（物体表面）
        zero_level = [0]
        if min_val <= 0 <= max_val:  # 只在范围包含0时绘制零等值线
            plt.contour(values_grid, levels=zero_level, colors='white', linewidths=2)
    except Exception as e:
        print(f"Warning: Could not generate contour lines: {e}")
    
    # 添加颜色条
    try:
        cbar = plt.colorbar(im, label='SDF Value')
    except Exception as e:
        print(f"Warning: Could not generate colorbar: {e}")
    
    # 设置标题和标签
    plt.title('SDF Slice Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 移除刻度标签（可选）
    plt.xticks([])
    plt.yticks([])
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='SDF Slice Visualizer',
        description='Create slice planes with SDF values for visualization')
    
    parser.add_argument("model", type=str, help="path to the model '.pt' file")
    parser.add_argument("-o", "--output-dir", type=str, default="meshes", help="output directory for generated files")
    parser.add_argument("-r", "--resolution", type=int, default=100, help="resolution of the slice plane")
    parser.add_argument("-s", "--size", type=float, default=1.0, help="size of the slice plane")
    parser.add_argument("-l", "--location", type=str, default="[0,0,0]", help="location of the slice plane center, format: [x,y,z]")
    parser.add_argument("-rot", "--rotation", type=str, default="[0,0,0]", help="rotation of the slice plane in radians, format: [rx,ry,rz]")
    parser.add_argument("-cpu", action="store_true", help="force CPU computation")
    parser.add_argument("-bs", "--batch-size", type=int, default=1024, help="batch size for model evaluation")
    parser.add_argument("-vmin", type=float, default=None, help="minimum value for normalization")
    parser.add_argument("-vmax", type=float, default=None, help="maximum value for normalization")
    parser.add_argument("-cmap", type=str, default="RdBu_r", help="matplotlib colormap for visualization")
    parser.add_argument("-png", action="store_true", help="generate PNG visualization")
    parser.add_argument("-dpi", type=int, default=300, help="DPI for PNG output")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析位置和旋转参数
    try:
        location = ast.literal_eval(args.location)
        if not isinstance(location, (list, tuple)) or len(location) != 3:
            raise ValueError("Location must be a tuple or list of length 3")
    except Exception as e:
        print(f"Error parsing location: {e}")
        print("Using default location [0, 0, 0]")
        location = [0, 0, 0]
    
    try:
        rotation = ast.literal_eval(args.rotation)
        if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
            raise ValueError("Rotation must be a tuple or list of length 3")
    except Exception as e:
        print(f"Error parsing rotation: {e}")
        print("Using default rotation [0, 0, 0]")
        rotation = [0, 0, 0]
    
    # 使用location和rotation参数
    print(f"Creating slice plane with location={location}, rotation={rotation}, size={args.size}")
    points, points_grid = create_plane_from_parameters(
        location=location,
        rotation=rotation, 
        size=args.size, 
        resolution=args.resolution
    )
    
    # 加载模型
    device = get_device(args.cpu)
    print("DEVICE:", device)
    model = load_model(args.model, device).to(device)
    
    # 评估SDF值
    print("Evaluating SDF on slice plane...")
    values = evaluate_sdf_on_plane(model, points, device, args.batch_size)
    values_grid = values.reshape(args.resolution, args.resolution)
    
    # 获取原始SDF值的范围
    print("Min SDF value:", values.min())
    print("Max SDF value:", values.max())
    
    # 归一化值（用于可视化）
    normalized_values = normalize_values(values, args.vmin, args.vmax)
    
    # 保存结果
    obj_path = os.path.join(args.output_dir, "slice_plane.obj")
    npy_path = os.path.join(args.output_dir, "slice_plane_scalar.npy")
    
    print(f"Saving slice plane to {obj_path}")
    num_vertices = save_slice_plane_obj(points_grid, obj_path)
    
    print(f"Saving SDF values to {npy_path}")
    # 注意：这里不再保存网格形状的数据，而是保存一维数组，与顶点顺序一致
    # 这样vertexScalarToUV函数可以直接使用这些值而不需要重塑
    np.save(npy_path, normalized_values)
    
    # 如果需要，生成PNG可视化
    if args.png:
        png_path = os.path.join(args.output_dir, "slice_visualization.png")
        visualize_sdf_slice(values_grid, png_path, args.vmin, args.vmax, args.cmap, args.dpi)
    
    # 保存平面参数供Blender使用
    params_path = os.path.join(args.output_dir, "slice_plane_params.npy")
    print(f"Saving slice plane parameters to {params_path}")
    np.savez(params_path, location=np.array(location), rotation=np.array(rotation), size=args.size)
    
    print("Done!")