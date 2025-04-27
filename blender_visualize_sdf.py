# # if you want to call the toolbox the old way with `blender -b -P demo_XXX.py`, then uncomment these two lines
# import sys, os
# sys.path.append("../../BlenderToolbox/")
import blendertoolbox as bt 
import bpy
import os
import numpy as np
from mathutils import Vector, Matrix
outputPath = os.path.abspath('./spot_slicePlane.png') 

## initialize blender
imgRes_x = 480 
imgRes_y = 480 
numSamples = 100 
exposure = 1.5 
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

## 归一化物体到0-1立方体中并居中
def normalize_object_to_unit_cube(obj):
    # 获取物体的边界框
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # 计算边界框的最小和最大坐标
    min_x = min(corner.x for corner in bbox_corners)
    max_x = max(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    min_z = min(corner.z for corner in bbox_corners)
    max_z = max(corner.z for corner in bbox_corners)
    
    # 计算边界框的尺寸和中心
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    # 计算缩放因子（使最大边缩放至1）
    max_size = max(size_x, size_y, size_z)
    scale_factor = 1.0 / max_size if max_size > 0 else 1.0
    
    # 缩放物体
    obj.scale *= scale_factor
    bpy.context.view_layer.update()
    
    # 重新计算边界框以获取新的中心
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center_x = sum(corner.x for corner in bbox_corners) / 8
    center_y = sum(corner.y for corner in bbox_corners) / 8
    center_z = sum(corner.z for corner in bbox_corners) / 8
    
    # 将物体移动到(0,0,0.5)
    obj.location.x += 0 - center_x
    obj.location.y += 0 - center_y
    obj.location.z += 0.5 - center_z
    bpy.context.view_layer.update()
    
    return obj

## read mesh (decoration for the scene)
location = (0, 0, 0)  # 使用原点作为初始加载位置
rotation = (0, 0, 0)  # 不旋转
scale = (1, 1, 1)     # 不缩放，后面会通过函数归一化
meshPath = 'data/spot.obj'
mesh = bt.readMesh(meshPath, location, rotation, scale)
normalize_object_to_unit_cube(mesh)  # 归一化物体到0-1立方体并居中到(0,0,0.5)

# 设置物体最终的位置和旋转
mesh.location = (1.12, -0.14, 0) 
rotation = (90, 0, 227)
scale = (1.5,1.5,1.5) 
x = rotation[0] * 1.0 / 180.0 * np.pi 
y = rotation[1] * 1.0 / 180.0 * np.pi 
z = rotation[2] * 1.0 / 180.0 * np.pi 
angle = (x, y, z)
mesh.rotation_euler = angle
bpy.ops.object.shade_smooth() 
bt.subdivision(mesh, level = 2)
meshColor = bt.colorObj(bt.derekBlue, 0.5, 1.0, 1.0, 0.0, 2.0)
bt.setMat_plastic(mesh, meshColor)

# 保存物体的变换信息，稍后用于切片平面
object_location = mesh.location.copy()
object_rotation = mesh.rotation_euler.copy()
object_scale = mesh.scale.copy()

# 生成SDF切片平面的法线方向
# 使用物体的旋转矩阵将[0,0,1]向量转换为物体坐标系中的方向
rotation_matrix = mesh.rotation_euler.to_matrix()
# 这里我们假设初始平面法线是Z轴方向[0,0,1]
initial_normal = Vector((0.5, 0.0, 1))
transformed_normal = rotation_matrix @ initial_normal
print(f"切片平面法线方向: {transformed_normal}")


# 在生成SDF切片数据之前，自动运行命令生成平面

normal_str = f"[{transformed_normal.x},{transformed_normal.y},{transformed_normal.z}]"
cmd = f'python visualize_sdf_slice.py output/spot/model_e300.pt -o meshes -r 200 -n "{normal_str}" -s 2.0 -png'
print(f"执行命令: {cmd}")

## 读取切片平面 - 现在切片平面与物体坐标系对齐
meshPath = 'meshes/slice_plane.obj'

# 在加载切片平面时，使用与物体相同的变换
slice_mesh = bt.readMesh(meshPath, object_location, (0,0,0), scale=object_scale*1.5)

# 应用与物体相同的旋转变换
slice_mesh.rotation_euler = object_rotation

# 可以根据需要微调切片平面位置
# slice_mesh.location.z += 0.0  # 调整切片平面沿局部Z轴偏移

bpy.ops.object.shade_flat() 
bt.subdivision(slice_mesh, level = 0)

# 加载切片平面的SDF值
scalar = np.load("meshes/slice_plane_scalar.npy")
print(scalar)
slice_mesh = bt.vertexScalarToUV(slice_mesh, scalar)

slice_mesh.data.uv_layers["funcUV"].active_render = True
            
useless = (0,0,0,1)
meshColor = bt.colorObj(useless, 0.5, 1.3, 1.0, 0.0, 0.4)
texturePath = 'meshes/RdBu_black.png' 
alpha = 0.75 # the smaller the more transparent
bt.setMat_texture(slice_mesh, texturePath, meshColor, alpha)

## set invisible plane (shadow catcher)
bt.invisibleGround(shadowBrightness=0.9)

## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
camLocation = (3, 0, 2)
lookAtLocation = (0,0,0.5)  # 注视物体中心
focalLength = 45 # (UI: click camera > Object Data > Focal Length)
cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

## set light
lightAngle = (6, -30, -155) 
strength = 2
shadowSoftness = 0.3
sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

## set ambient light
bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

## set gray shadow to completely white with a threshold 
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

## save blender file so that you can adjust parameters in the UI
bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

## save rendering
bt.renderImage(outputPath, cam)