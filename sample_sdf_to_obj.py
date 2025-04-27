import argparse
import numpy as np
import torch
import os
from common.models import load_model
from common.utils import get_device
import colorsys

def sample_sdf_in_volume(model, resolution, device, batch_size=1024):
    """
    在-0.5~0.5的空间内按网格采样SDF值
    
    参数:
    model: SDF模型
    resolution: 每个维度的采样分辨率
    device: 计算设备
    batch_size: 批处理大小
    
    返回:
    points: 采样点坐标 (resolution^3, 3)
    values: 对应的SDF值 (resolution^3,)
    """
    # 固定在-0.5~0.5范围内创建网格点
    x = np.linspace(-0.5, 0.5, resolution)
    y = np.linspace(-0.5, 0.5, resolution)
    z = np.linspace(-0.5, 0.5, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 重塑为点列表
    points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # 评估SDF值
    model.eval()
    values = np.zeros(points.shape[0])
    
    with torch.no_grad():
        for i in range(0, points.shape[0], batch_size):
            batch_points = torch.tensor(points[i:i+batch_size], dtype=torch.float32).to(device)
            batch_values = model(batch_points)
            values[i:i+batch_size] = batch_values.cpu().numpy().flatten()
    
    return points, values

def map_sdf_to_color(value, vmin, vmax, colormap='rdbu'):
    """
    将SDF值映射到颜色
    
    参数:
    value: SDF值
    vmin, vmax: 值范围
    colormap: 颜色映射方案
    
    返回:
    color: RGB颜色值 [r, g, b]
    """
    # 将值归一化到 [0, 1] 范围
    normalized = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    normalized = np.clip(normalized, 0, 1)
    
    if colormap == 'rdbu':
        # 红蓝映射: 负值为红色，正值为蓝色，零值为白色
        if value < 0:  # 负值: 红色到白色
            t = 1 - normalized * 2  # 将 [0, 0.5] 映射到 [1, 0]
            r = 1.0
            g = b = t
        else:  # 正值: 白色到蓝色
            t = (normalized - 0.5) * 2  # 将 [0.5, 1] 映射到 [0, 1]
            b = 1.0
            r = g = 1.0 - t
    elif colormap == 'rainbow':
        # 彩虹映射: 使用HSV颜色空间
        h = normalized * 0.7  # 从红到紫 (hue: 0到0.7)
        s = 1.0
        v = 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
    else:  # 默认灰度
        r = g = b = 1.0 - normalized
    
    return [r, g, b]

def compute_radius_from_sdf(values):
    """
    直接使用SDF值作为半径，内部点(负值)用绝对值
    
    参数:
    values: SDF值数组
    
    返回:
    radius_values: 半径数组
    """
    # 复制SDF值
    radius_values = values.copy()
    
    # 对于内部点(负值)，使用绝对值
    radius_values[values < 0] = np.abs(values[values < 0])
    
    return radius_values

def save_points_to_ply(points, values, output_path, colormap='rdbu'):
    """
    将点保存到PLY文件，包含颜色和半径信息
    
    参数:
    points: 点坐标
    values: 对应的SDF值
    output_path: 输出PLY文件路径
    colormap: 颜色映射方案
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 计算每个点的半径 - 直接使用SDF值，内部点用绝对值
    radius_values = compute_radius_from_sdf(values)
    
    # 为颜色映射准备值范围
    v_min, v_max = values.min(), values.max()
    
    # 计算并准备所有颜色值（转换为0-255整数范围）
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for i in range(len(points)):
        color_float = map_sdf_to_color(values[i], v_min, v_max, colormap)
        colors[i, 0] = int(color_float[0] * 255)  # R
        colors[i, 1] = int(color_float[1] * 255)  # G
        colors[i, 2] = int(color_float[2] * 255)  # B
    
    # 打开文件进行写入，使用UTF-8编码
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入PLY头部信息
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment SDF Point Cloud Visualization\n")
        f.write(f"comment SDF Value Range: [{v_min:.6f}, {v_max:.6f}]\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float radius\n")
        f.write("end_header\n")
        
        # 写入点、颜色、半径和SDF值
        for i in range(len(points)):
            point = points[i]
            color = colors[i]
            radius = radius_values[i]

            
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
            f.write(f"{color[0]} {color[1]} {color[2]} ")
            f.write(f"{radius:.6f}\n")
         
    
    print(f"Saved {len(points)} points to {output_path}")
    print(f"Each point includes position, color, radius and SDF value")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='SDF Points to PLY',
        description='Sample SDF points and save them as points with radius in a PLY file')
    
    parser.add_argument("model", type=str, help="path to the model '.pt' file")
    parser.add_argument("-o", "--output", type=str, default="meshes/sdf_points.ply", help="output PLY file path")
    parser.add_argument("-r", "--resolution", type=int, default=20, help="sampling resolution per dimension")
    parser.add_argument("-cm", "--colormap", type=str, default="rdbu", choices=["rdbu", "rainbow", "gray"], help="color map for SDF values")
    parser.add_argument("-cpu", action="store_true", help="force CPU computation")
    parser.add_argument("-bs", "--batch-size", type=int, default=1024, help="batch size for model evaluation")
    
    args = parser.parse_args()
    
    # 修改默认输出扩展名为ply
    if args.output.endswith('.obj'):
        args.output = args.output[:-4] + '.ply'
    elif not args.output.endswith('.ply'):
        args.output = args.output + '.ply'
    
    # 加载模型
    device = get_device(args.cpu)
    print("Device:", device)
    model = load_model(args.model, device).to(device)
    
    # 采样SDF值（固定在-0.5~0.5范围内）
    print(f"Sampling SDF values in volume [-0.5,-0.5,-0.5] to [0.5,0.5,0.5] with resolution {args.resolution}...")
    points, values = sample_sdf_in_volume(
        model, 
        resolution=args.resolution,
        device=device,
        batch_size=args.batch_size
    )
    
    print(f"Total points: {len(points)}")
    
    # 输出SDF值的基本统计信息
    print(f"SDF value statistics:")
    print(f"  Min: {values.min()}")
    print(f"  Max: {values.max()}")
    print(f"  Mean: {values.mean()}")
    print(f"  Std: {values.std()}")
    
    # 将点保存为PLY文件
    print(f"Saving points to PLY file: {args.output}")
    save_points_to_ply(
        points, 
        values, 
        args.output,
        colormap=args.colormap
    )
    
    print("Done!")