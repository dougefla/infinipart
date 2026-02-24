# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Lingjie Mei
from collections.abc import Callable, Iterable
import glob
import importlib
import json
import os
from pathlib import Path
from math import radians
import shutil
import subprocess

import gin
from mathutils import Matrix

import bpy
import numpy as np
#import torch
import trimesh
from mathutils import Vector
from tqdm import tqdm

# from pytorch3d.io import load_obj
# from pytorch3d.structures import Meshes
# from pytorch3d.ops import sample_points_from_meshes
# from bpy_lib import *
# from pytorch3d.io import load_ply, save_ply
#from infinigen.assets.materials import common
from infinigen.core import surface, tagging
import infinigen.core.util.blender as butil
from infinigen.assets.utils.decorate import read_co, read_material_index, write_co, write_material_index
from infinigen.core.util.blender import select_none

import urdfpy
from infinigen.tools import export
import infinigen.assets.utils.usdutils as usdutils
import math

# def sample_and_save_points(verts, faces, path, num_samples=50000, return_normals=True):

#     meshes = Meshes(verts=[verts], faces=[faces])
#     samples, normals = sample_points_from_meshes(
#         meshes,
#         num_samples = num_samples,
#         return_normals = return_normals,
#         return_textures = False,
#     )

#     samples = samples.squeeze(0)
#     normals = normals.squeeze(0)
#     np.savez(f'{path}.npz', points=samples.cpu().numpy(), normals=normals.cpu().numpy())


# def write_json(data_path, json_path, idx=None, names=None, category=None):
#     # case_list = os.listdir(data_path)
#     # case_list_int = [int(case) for case in case_list]
#     # case_list_int.sort()
#     # case_list =[str(case) for case in case_list_int]
#     # for case in tqdm(case_list):
#     infos = []
#     info_case = {}
#     info_case["id"] = idx

#     for i, name in enumerate(names):
#         obj_name = name

#         info_case["obj_name"] = obj_name
#         info_case["category"] = category
#         info_case["file_obj_path"] = os.path.join(data_path, f"{idx}/objs/whole.obj")
#         info_case["file_pcd_path"] = os.path.join(
#             data_path, f"{idx}/point_cloud/whole.npz"
#         )
#         info_parts = []

#         for result_part_info in result_parts_info:
#             part_info = {}
#             part_info["part_name"] = obj_name + "_part"
#             part_info["file_name"] = str(result_part_info["id"]) + ".obj"
#             part_info["file_obj_path"] = os.path.join(
#                 data_path, f"{idx}/objs/{str(result_part_info['id'])}.obj"
#             )
#             part_info["file_pcd_path"] = os.path.join(
#                 data_path, f"{idx}/point_cloud/{str(result_part_info['id'])}.npz"
#             )
#             info_parts.append(part_info)
#         info_case["part"] = info_parts
#         infos.append(info_case)

#     with open(f"{json_path}/data_infos.json", "w") as f:
#         json.dump(infos, f, indent=2)

robot_tree = {}
root = None
internal_bbox = None

uuid = 0
def get_uuid():
    global uuid
    uuid += 1
    return uuid

def get_joint_name(type):
    return f"joint_{type}_{get_uuid()}" 

def get_link_name(name):
    return f"link_{name}_{get_uuid()}"


# bbox here is a list of 6 values, [x, x_, y, y_, z, z_] where x, x_ are the x coordinates of the left and right faces, y, y_ are the y coordinates of the front and back faces, z, z_ are the z coordinates of the top and bottom faces
def add_internal_bbox(bbox):
    global internal_bbox
    if internal_bbox is None:
        internal_bbox = []
    internal_bbox.append(bbox)

def center(obj):
    return (Vector(obj.bound_box[0]) + Vector(obj.bound_box[-2])) * obj.scale / 2.0


def origin2lowest(obj, vertical=False, centered=False, approximate=False):
    co = read_co(obj)
    if not len(co):
        return
    i = np.argmin(co[:, -1])
    if approximate:
        indices = np.argsort(co[:, -1])
        obj.location = -np.mean(co[indices[: len(co) // 10]], 0)
        obj.location[-1] = -co[i, -1]
    elif centered:
        obj.location = -center(obj)
        obj.location[-1] = -co[i, -1]
    elif vertical:
        obj.location[-1] = -co[i, -1]
    else:
        obj.location = -co[i]
    butil.apply_transform(obj, loc=True)


def origin2highest(obj):
    co = read_co(obj)
    i = np.argmax(co[:, -1])
    obj.location = -co[i]
    butil.apply_transform(obj, loc=True)


def origin2leftmost(obj):
    co = read_co(obj)
    i = np.argmin(co[:, 0])
    obj.location = -co[i]
    butil.apply_transform(obj, loc=True)


def data2mesh(vertices=(), edges=(), faces=(), name=""):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()
    return mesh


def mesh2obj(mesh):
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def trimesh2obj(trimesh):
    obj = butil.object_from_trimesh(trimesh, "")
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def obj2trimesh(obj):

# set the new_obj as active object for later process
    bpy.context.view_layer.objects.active = obj

# make sure new_obj has single user copy
    bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)
    butil.modify_mesh(obj, "TRIANGULATE", min_vertices=3)
    vertices = read_co(obj)
    arr = np.zeros(len(obj.data.polygons) * 3)
    obj.data.polygons.foreach_get("vertices", arr)
    faces = arr.reshape(-1, 3)
    return trimesh.Trimesh(vertices, faces)


def obj2trimesh_and_save(obj, path=None, idx="unknown", is_return=False):
    trimesh_object = obj2trimesh(obj)
    if idx == "unknown":
        idx = f"random_{np.random.randint(0, 10000)}"
    else:
        idx = str(idx)
    if path is None:
        path = os.path.join(os.path.curdir, "outputs")
    if not os.path.exists(path):
        os.mkdir(path)
    if not isinstance(path, Path):
        path = Path(path)
    (path / ("parts_seed_" + idx)).mkdir(exist_ok=True)
    # File path for saving the .obj file
    file_path = os.path.join(
        path, "parts_seed_" + idx, f"{obj.name}_part_{int(idx)+1}.obj"
    )
    trimesh_object.export(file_path)
    if is_return:
        return trimesh_object


def obj2trimesh_and_save_normalized(
    obj,
    path=None,
    idx="unknown",
    name=None,
    is_return=False,
    point_cloud=False,
    export_json=True,
):
    trimesh_object = obj2trimesh(obj)
    if idx == "unknown":
        idx = f"random_{np.random.randint(0, 10000)}"
    else:
        idx = str(idx)
    if path is None:
        path = os.path.join(os.path.curdir, "outputs")
    if not os.path.exists(path):
        os.mkdir(path)
    if not isinstance(path, Path):
        path = Path(path)
    (path / idx).mkdir(exist_ok=True)
    (path / idx / "objs").mkdir(exist_ok=True)
    # (path / idx / "point_cloud").mkdir(exist_ok=True)
    # File path for saving the .obj file
    file_path = os.path.join(path, idx, f"{obj.name}_part_{int(idx)+1}.obj")
    # normalize trimesh
    # Get the current bounds of the mesh
    bounds = trimesh_object.bounds

    # Calculate the range of the current bounds
    x_range = bounds[1][0] - bounds[0][0]
    y_range = bounds[1][1] - bounds[0][1]
    z_range = bounds[1][2] - bounds[0][2]

    # Calculate the scale factor to fit the mesh within [-1, 1]
    scale_factor = 1.0 / max(x_range, y_range, z_range)

    # Scale the mesh
    trimesh_object.vertices *= 2 * scale_factor

    # Translate the mesh to center it around the origin
    trimesh_object.vertices -= [
        scale_factor * (bounds[0][i] + bounds[1][i]) for i in range(3)
    ]

    # Now the vertices should be within the range [-1, 1]
    trimesh_object.export(file_path)

    # if point_cloud:
    #     pc_path = os.path.join(path, idx, f"{idx}.npz")
    # sample_and_save_points(torch.tensor(trimesh_object.vertices, dtype=torch.float32), torch.tensor(trimesh_object.faces), pc_path)

    if is_return:
        return trimesh_object


def new_cube(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.mesh.primitive_cube_add(**kwargs)
    return bpy.context.active_object


def new_bbox(x, x_, y, y_, z, z_):
    obj = new_cube()
    obj.location = (x + x_) / 2, (y + y_) / 2, (z + z_) / 2
    obj.scale = (x_ - x) / 2, (y_ - y) / 2, (z_ - z) / 2
    butil.apply_transform(obj, True)
    return obj


def new_bbox_2d(x, x_, y, y_, z=0):
    obj = new_plane()
    obj.location = (x + x_) / 2, (y + y_) / 2, z
    obj.scale = (x_ - x) / 2, (y_ - y) / 2, 1
    butil.apply_transform(obj, True)
    return obj


def new_icosphere(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.mesh.primitive_ico_sphere_add(**kwargs)
    return bpy.context.active_object


def new_circle(**kwargs):
    kwargs["location"] = kwargs.get("location", (1, 0, 0))
    bpy.ops.mesh.primitive_circle_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_base_circle(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.mesh.primitive_circle_add(**kwargs)
    obj = bpy.context.active_object
    return obj


def new_empty(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.object.empty_add(**kwargs)
    obj = bpy.context.active_object
    obj.scale = kwargs.get("scale", (1, 1, 1))
    return obj


def new_plane(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.mesh.primitive_plane_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_cylinder(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0.5))
    kwargs["depth"] = kwargs.get("depth", 1)
    bpy.ops.mesh.primitive_cylinder_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_base_cylinder(**kwargs):
    bpy.ops.mesh.primitive_cylinder_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_grid(**kwargs):
    kwargs["location"] = kwargs.get("location", (0, 0, 0))
    bpy.ops.mesh.primitive_grid_add(**kwargs)
    obj = bpy.context.active_object
    butil.apply_transform(obj, loc=True)
    return obj


def new_line(subdivisions=1, scale=1.0):
    vertices = np.stack(
        [
            np.linspace(0, scale, subdivisions + 1),
            np.zeros(subdivisions + 1),
            np.zeros(subdivisions + 1),
        ],
        -1,
    )
    edges = np.stack([np.arange(subdivisions), np.arange(1, subdivisions + 1)], -1)
    obj = mesh2obj(data2mesh(vertices, edges))
    return obj


def join_objects(obj):
    butil.select_none()
    if not isinstance(obj, list):
        obj = [obj]
    if len(obj) == 1:
        return obj[0]
    bpy.context.view_layer.objects.active = obj[0]
    butil.select_none()
    butil.select(obj)
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.location = 0, 0, 0
    obj.rotation_euler = 0, 0, 0
    obj.scale = 1, 1, 1
    butil.select_none()
    return obj


def add_joint(parent, child, joint_info):
    robot_tree[child] = (parent, joint_info)


def join_objects_save_whole(obj, path=None, idx="unknown", name=None, join=True, use_bpy=False):
    butil.select_none()
    if not isinstance(obj, list):
        obj = [obj]
    bpy.context.view_layer.objects.active = obj[0]
    butil.select_none()
    butil.select(obj)
    if join:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.location = 0, 0, 0
    obj.rotation_euler = 0, 0, 0
    obj.scale = 1, 1, 1
    save_whole_object_normalized(obj, path, idx, use_bpy=use_bpy)
    butil.select_none()
    return obj


def save_parts_join_objects(obj, path=None, idx="unknown", name=None):
    butil.select_none()
    if not isinstance(obj, list):
        obj = [obj]
    if name is None:
        name = "unknown"
    if not isinstance(name, list):
        name = [name] * len(obj)
    if len(obj) == 1:
        return obj[0]
    # The original render engine and world node_tree should be memorized
    original_render_engine = bpy.context.scene.render.engine
    original_node_tree = bpy.context.scene.world.node_tree
    #Save a reference to the original scene
    original_scene = bpy.context.scene
    save_parts(obj, path, idx, name)
    # We need to link all these objects into view_layer
    view_layer = bpy.context.view_layer
    for part in obj:
        if part.name not in view_layer.objects:
            view_layer.active_layer_collection.collection.objects.link(part)
    bpy.context.scene.render.engine = original_render_engine
    # Now switch back to the original scene
    bpy.context.window.scene = original_scene
    bpy.context.view_layer.objects.active = obj[0]
    butil.select_none()
    butil.select(obj)
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.location = 0, 0, 0
    obj.rotation_euler = 0, 0, 0
    obj.scale = 1, 1, 1
    butil.select_none()
    return obj


def save_parts(objects, path=None, idx="unknown", name=None):
    assert len(objects) == len(name)
    if idx == "unknown":
        idx = f"random_{np.random.randint(0, 10000)}"
    else:
        idx = str(idx)
    if path is None:
        path = os.path.join(os.path.curdir, "outputs")
    if not os.path.exists(path):
        os.makedirs(path)
    if not isinstance(path, Path):
        path = Path(path)
    (path / (idx)).mkdir(exist_ok=True)
    butil.select_none()
    for i, part in enumerate(objects):
        # Reference the current view layer
        view_layer = bpy.context.view_layer
        # Link the object to the active view layer's collection
        if part.name not in view_layer.objects:
            view_layer.active_layer_collection.collection.objects.link(part)
        bpy.ops.object.select_all(action="DESELECT")
        # Select the current object
        part.select_set(True)
        # Create a new scene
        new_scene = bpy.data.scenes.new(f"Scene_for_{part.name}")
        # Link the object to the new scene
        new_scene.collection.objects.link(part)
        # Make the new scene active
        bpy.context.window.scene = new_scene
        # File path for saving the .blend file
        if name:
            file_path = os.path.join(
                path, "parts_seed_" + idx, f"{name[i]}_part_{i+1}.blend"
            )
        else:
            file_path = os.path.join(
                path, "parts_seed_" + idx, f"{part.name}_part_{i+1}.blend"
            )
        # Save the current scene as a new .blend file
        bpy.ops.wm.save_as_mainfile(filepath=file_path)
    butil.select_none()


def save_whole_object(object, path=None, idx="unknown"):
    if idx == "unknown":
        idx = f"random_{np.random.randint(0, 10000)}"
    else:
        idx = str(idx)
    if path is None:
        path = os.path.join(os.path.curdir, "outputs")
    if not os.path.exists(path):
        os.makedirs(path)
    if not isinstance(path, Path):
        path = Path(path)
    (path / (idx)).mkdir(exist_ok=True)
    butil.select_none()
    # # Reference the current view layer
    # view_layer = bpy.context.view_layer
    # # Link the object to the active view layer's collection
    # if object.name not in view_layer.objects:
    #     view_layer.active_layer_collection.collection.objects.link(object)
    # bpy.ops.object.select_all(action="DESELECT")
    # Select the current object
    object.select_set(True)
    # # Create a new scene
    # new_scene = bpy.data.scenes.new(f"Scene_for_{object.name}")
    # # Link the object to the new scene
    # new_scene.collection.objects.link(object)
    # # Make the new scene active
    # bpy.context.window.scene = new_scene
    # # File path for saving the .blend file
    file_path = os.path.join(path, idx, "objs/whole.obj")
    # Save the current scene as a new .obj file
    trimesh_object = obj2trimesh(object)
    trimesh_object.export(file_path)
    # Save the current scene as a new .blend file
    # bpy.ops.wm.save_as_mainfile(filepath=file_path)
    butil.select_none()

import random

def get_translation_matrix(x, y, z):
    matrix = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    return matrix

# def setup_camera():
#     cam_dist = 6
    
#     camera = bpy.context.active_object
#     camera.data.sensor_height = (
#         camera.data.sensor_width * scene.render.resolution_y / scene.render.resolution_x
#     )
#     for area in bpy.context.screen.areas:
#         if area.type == "VIEW_3D":
#             area.spaces.active.region_3d.view_perspective = "CAMERA"
#             break
#     cam_info_ng = bpy.data.node_groups.get("nodegroup_active_cam_info")
#     if cam_info_ng is not None:
#         cam_info_ng.nodes["Object Info"].inputs["Object"].default_value = camera
#     return camera, camera.parent

def render_object_texture_and_save(obj, material, path):
    if material is None:
        return
    name = material.name
    r = subprocess.call(['python' ,'-m' ,'infinigen_examples.generate_individual_assets' ,'--output_folder' ,path ,'-f' ,name ,'-n' ,'1'])
    # template = None
    # for subdir in os.listdir("infinigen/assets/materials"):
    #     with gin.unlock_config():
    #         module = importlib.import_module(
    #         f'infinigen.assets.materials.{subdir.split(".")[0]}'
    #         )
    #         if hasattr(module, name) and hasattr(module, 'apply'):
    #             template = getattr(module, 'apply')
    #             break
    # if template is None:
    #     return
    # #template(obj, material)
        
    # # create new scene
    # original_scene = bpy.context.scene
    # scene = bpy.data.scenes.new("temp_scene")
    # scene.render.engine = "CYCLES"
    # scene.render.resolution_x, scene.render.resolution_y = 1024, 1024
    # scene.cycles.samples = 200
    # bpy.context.window.scene = scene
    # bpy.ops.object.camera_add(location=(0, -6, 0), rotation=(np.pi / 2, 0, 0))
    # camera = bpy.context.active_object
    # scene.camera = camera
    # #bpy.ops.mesh.primitive_plane_add(size=100)
    # #obj = bpy.context.active_object
    # #template(obj)
     
    # light_data = bpy.data.lights.new(name='Direct_Light', type='POINT')
 
    # # 将光源对象转换为方向光类型
    # light_data.type = 'SUN'
    # light_data.energy = 200
    # # 将方向光源添加到场景中
    # light_object = bpy.data.objects.new(name='My Directional Light', object_data=light_data)
    # scene.collection.objects.link(light_object)
    # light_object.location = (-10, -10, 10)
 
    # # 设置方向光源的方向
    # # light_object.rotation_euler = Vector((0, 0, 0))
    # # light_object.rotation_euler.rotate_axis('X', 1.5708)  # 90度弧度值
 
    # scene.render.filepath = str(path)
    # #add camera
    # bpy.ops.render.render(write_still=True)
    # bpy.context.window.scene = original_scene 
saved_objs = []
def save_whole_object_normalized(object, path=None, idx="unknown", name=None, use_bpy=False):
    global saved_objs
    global robot_tree, root
    global internal_bbox
    big_obj = join_objects(saved_objs)
    save_obj_parts_add([big_obj], path, idx, name, first=False, use_bpy=True, parent_obj_id="")
    if idx == "unknown":
        idx = f"random_{np.random.randint(0, 10000)}"
    else:
        idx = str(idx)
    if path is None:
        path = os.path.join(os.path.curdir, "outputs")
    if not os.path.exists(path):
        os.makedirs(path)
    if not isinstance(path, Path):
        path = Path(path)
    (path / (idx)).mkdir(exist_ok=True)
    (path / (idx) / "objs").mkdir(exist_ok=True)
    # (path / (idx) / "point_cloud").mkdir(exist_ok=True)
    json_path = os.path.join(path, f"data_infos_{idx}.json")
    if not os.path.exists(json_path):
        infos = []
    else:
        with open(json_path, "r") as f:
            infos = json.load(f)

    if infos:
        info_case = infos[-1]
    else:
        info_case = {}

    if "id" not in info_case.keys():
        info_case["id"] = idx
    obj_name = os.path.basename(path)[:-7]
    info_case["obj_name"] = obj_name
    # info_case["category"] = category
    info_case["file_obj_path"] = os.path.join(path, f"{idx}/objs/whole.obj")
    butil.select_none()
    # # Reference the current view layer
    # view_layer = bpy.context.view_layer
    # # Link the object to the active view layer's collection
    # if object.name not in view_layer.objects:
    #     view_layer.active_layer_collection.collection.objects.link(object)
    # bpy.ops.object.select_all(action="DESELECT")
    # Select the current object
    object.select_set(True)
    # # Create a new scene
    # new_scene = bpy.data.scenes.new(f"Scene_for_{object.name}")
    # # Link the object to the new scene
    # new_scene.collection.objects.link(object)
    # # Make the new scene active
    # bpy.context.window.scene = new_scene
    # # File path for saving the .blend file
    file_path = os.path.join(path, idx, "objs", "whole.obj")
    # Save the current scene as a new .obj file
    trimesh_object = obj2trimesh(object)
    # normalize trimesh
    # Get the current bounds of the mesh
    #bounds = trimesh_object.bounds.copy()

    # Calculate the range of the current bounds
    #x_range = bounds[1][0] - bounds[0][0]
    #y_range = bounds[1][1] - bounds[0][1]
    #z_range = bounds[1][2] - bounds[0][2]

    # Calculate the scale factor to fit the mesh within [-1, 1]
    #scale_factor = 1.0 / max(x_range, y_range, z_range)

    # Translate the mesh to center it around the origin
    # trimesh_object.vertices -= [
    #     (bounds[0][j] + bounds[1][j]) / 2 for j in range(3)
    # ]

    # Scale the mesh
    #trimesh_object.vertices *= scale_factor

    angle_degrees = 90  # 45 degrees
    angle_radians = np.radians(angle_degrees)

    # # Create a rotation matrix for the x-axis
    # # For a clockwise rotation when viewed from the positive x-axis,
    # # we use a positive angle with np.cos and a negative angle with np.sin
    rotation_matrix = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )

    # # Apply the rotation matrix to the vertices
    rotated_vertices = np.dot(trimesh_object.vertices, rotation_matrix.T)

    # Create a new Trimesh object with the rotated vertices
    trimesh_object = trimesh.Trimesh(
        vertices=rotated_vertices, faces=trimesh_object.faces
    )

    #input()

    # sample point clouds
    # sample_and_save_points(torch.tensor(trimesh_object.vertices, dtype=torch.float32), torch.tensor(trimesh_object.faces), os.path.join(path, f"{idx}/point_cloud/whole"))
    #path_list = glob.glob(os.path.join(path, idx, "objs", "*.obj"))
    path_list = []
    obj_dirs = os.listdir(os.path.join(path, idx, "objs"))
    for obj_dir in obj_dirs:
        path_single_obj = os.path.join(path, idx, "objs", obj_dir)
        if os.path.isdir(path_single_obj):
            if os.path.exists(os.path.join(path_single_obj, "textures")):
                 shutil.rmtree(os.path.join(path_single_obj, "textures"))
            dir = os.listdir(path_single_obj)
            # if "textures" in dir:
            #     dir.remove("textures")
            dir = dir[0]
            dir = os.path.join(path_single_obj, dir)
            all_files = os.listdir(dir)
            for file in all_files:
                if not file.endswith(".obj") and not file.endswith(".mtl"):
                    #print(os.path.join(path_single_obj, file), os.path.join(dir, file))
                    shutil.copyfile(os.path.join(dir, file), os.path.join(path_single_obj, file))
                elif file.endswith(".obj"):
                    # mesh_idx = int(obj_dir)
                    # link = int(obj_dir)
                    # # if robot_tree[link][1] is not None and robot_tree[link][1].get("substitute_mesh_idx", None) is not None:
                    # #     mesh_idx = robot_tree[link][1]["substitute_mesh_idx"]
                    # real_path = os.path.join(path, idx, "objs", str(mesh_idx))
                    # f = [f for f in os.listdir(real_path) if f != 'textures' and os.path.isdir(os.path.join(real_path, f))]
                    # if len(f) == 0:
                    #     real_path = os.path.join(real_path, f"{mesh_idx}.obj")
                    # else:
                    #     f = f[0]
                    #     real_path = os.path.join(real_path, f)
                    #     real_path = os.path.join(real_path, f"{f}.obj")
                    shutil.copyfile(os.path.join(dir, file), os.path.join(path_single_obj, obj_dir + '.obj'))
                    path_list.append(os.path.join(path_single_obj, obj_dir + '.obj'))
                    with open(os.path.join(path_single_obj, obj_dir + '.obj'), 'r') as f:
                        lines = f.readlines()
                        lines[1] = 'mtllib ' + obj_dir + '.mtl\n'
                        lines[2] = 'usemtl ' + obj_dir + '\n'
                    with open(os.path.join(path_single_obj, obj_dir + '.obj'), 'w') as f:
                        f.writelines(lines)
                else:
                    shutil.copyfile(os.path.join(dir, file), os.path.join(path_single_obj, obj_dir + '.mtl'))
                    with open(os.path.join(path_single_obj, obj_dir + '.mtl'), 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith('map_Kd') or line.startswith('map_Pr') or line.startswith('map_Bump'):
                                items = line.split(' ')
                                path_ = items[-1].strip()
                                path_ = path_.split('/')[-1]
                                items[-1] = path_
                                line = ' '.join(items)
                    with open(os.path.join(path_single_obj, obj_dir + '.mtl'), 'w') as f:
                        f.writelines(lines)
            shutil.rmtree(dir)

    #path_list.remove(file_path)

    x_min, y_min, z_min, x_max, y_max, z_max = 300, 300, 300, -300, -300, -300
    origins = {}
    for obj_path in path_list:
        index = int(str(obj_path).split('/')[-1].split('.')[0])
        mesh_idx = index
        if robot_tree.get(index, None) is not None and robot_tree[index][1] is not None and robot_tree[index][1].get("substitute_mesh_idx", None) is not None:
             mesh_idx = robot_tree[index][1]["substitute_mesh_idx"]
        mesh_src_path = os.path.join(path, idx, "objs", f"{mesh_idx}", f"{mesh_idx}.obj")
        butil.select_none()
        bpy.ops.wm.obj_import(filepath=obj_path)
        obj = bpy.context.active_object
        co = read_co(obj)
        co = np.dot(co, rotation_matrix.T)
        bs = [
            [co[:, 0].min(), co[:, 1].min(), co[:, 2].min()],
            [co[:, 0].max(), co[:, 1].max(), co[:, 2].max()],
        ]
        # bs = mesh.bounds
        origins[index] = ((bs[0][0] + bs[1][0]) / 2, (bs[0][1] + bs[1][1]) / 2, (bs[0][2] + bs[1][2]) / 2)
        # x_min, y_min, z_min, x_max, y_max, z_max = min(x_min, bs[0][0]), min(y_min, bs[0][1]), min(z_min, bs[0][2]), max(x_max, bs[1][0]), max(y_max, bs[1][1]), max(z_max, bs[1][2])
        co[:, 0] -= origins[index][0]
        co[:, 1] -= origins[index][1]
        co[:, 2] -= origins[index][2]
        write_co(obj, co)
        butil.apply_transform(obj, loc=True)
        obj.name = str(index)
        # if mesh_idx != index:
        #     butil.select_none()
        #     bpy.ops.wm.obj_import(filepath=mesh_src_path)
        #     obj = bpy.context.active_object
        #     obj.name = str(index)
        bpy.ops.wm.obj_export(
                filepath=obj_path,
                export_colors=True,
                export_eval_mode="DAG_EVAL_RENDER",
                export_selected_objects=True,
                export_pbr_extensions=False,
                export_materials=True,
                export_normals = True,
                apply_modifiers = True
            )
        # bpy.ops.export_scene.fbx(
        #         filepath=obj_path.split('.')[0] + '.fbx',
        #         path_mode="COPY",
        #         embed_textures=True,
        #         use_selection=True,
        #     )
        with open(str(obj_path).replace('.obj', '.mtl'), 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith('map_Kd') or line.startswith('map_Pr') or line.startswith('map_Bump'):
                                items = line.split(' ')
                                path_ = items[-1].strip()
                                path_ = path_.split('/')[-1]
                                items[-1] = path_
                                line = ' '.join(items)
        with open(str(obj_path).replace('.obj', '.mtl'), 'w') as f:
                        f.writelines(lines)
        # bpy.ops.wm.usd_export(
        #     filepath=obj_path.split('.')[0] + '.usd',
        #     export_textures=True,
        #     use_instancing=True,
        #     # overwrite_textures=True,
        #     selected_objects_only=True,
        #     root_prim_path="/World",
        #     export_materials=True,
        #     #generate_materialx_network=True
        # )

        
        #os.remove(obj_path)
        #obj_path = obj_path.split('.')[0] + '.stl'
        #bpy.ops.export_mesh.stl(filepath=obj_path, use_selection=True)
        # mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        #mesh.export(obj_path)
        # with open(os.path.join(obj_path), 'r') as f:
        #                 lines = f.readlines()
        #                 lines.insert(1, 'mtllib ' + obj_path.split('/')[-1].split('.')[0] + '.mtl\n')
        # os.remove(os.path.join(obj_path))
        # with open(os.path.join(obj_path), 'w') as f:
        #                 f.writelines(lines)
    #usdutils.init_usd_stage(os.path.join(path, idx, "scene.usd"))
    links = {}
    joints= []
    origins["world"] = (0, 0, 0)
    # Save per-part origins (centroids before centering) for downstream processing
    origins_path = os.path.join(path, idx, "origins.json")
    _origins_ser = {str(k): list(v) for k, v in origins.items()}
    with open(origins_path, "w") as _f:
        json.dump(_origins_ser, _f, indent=2)
    print(f"[origins] Saved {len(origins)} origins to {origins_path}")
    print(robot_tree)

    root = urdfpy.Link("l_world", visuals=None, collisions=None, inertial=None)
    links["l_world"] = root
    for link in robot_tree.keys():
        if f"l_{link}" not in links.keys():
            mesh_idx = link
            if robot_tree[link][1] is not None and robot_tree[link][1].get("substitute_mesh_idx", None) is not None:
                mesh_idx = robot_tree[link][1]["substitute_mesh_idx"]
            if mesh_idx != link:
                shutil.rmtree(os.path.join(path, idx, "objs", f"{link}"))
                shutil.copytree(os.path.join(path, idx, "objs", f"{mesh_idx}"), os.path.join(path, idx, "objs", f"{link}"))
                for file in os.listdir(os.path.join(path, idx, "objs", f"{link}")):
                    if not file.endswith('.mtl') and file.startswith(f"{mesh_idx}"):
                        os.rename(os.path.join(path, idx, "objs", f"{link}", file), os.path.join(path, idx, "objs", f"{link}", file.replace(f"{mesh_idx}", f"{link}")))
            material = None
            if os.path.isfile(os.path.join(path, idx, "objs", f"{mesh_idx}.png")):
                texture = urdfpy.Texture(filename=os.path.join(path, idx, "objs", f"{mesh_idx}.png"))
                material = urdfpy.Material(name=get_link_name("material"), texture=texture)
            collision = None#[urdfpy.Collision(name="temp", origin=None, geometry=urdfpy.Geometry(mesh=urdfpy.Mesh(filename=os.path.join(path, idx, "objs", f"{mesh_idx}",f"{mesh_idx}.obj"))))]
            l = urdfpy.Link(f'l_{link}', visuals=[urdfpy.Visual(material=material, geometry=urdfpy.Geometry(mesh=urdfpy.Mesh(filename=os.path.join(path, idx, "objs",f"{mesh_idx}", f"{mesh_idx}.obj"))))], collisions=collision, inertial=None)
            links[f"l_{link}"] = l
            #usdutils.add_mesh(os.path.join(path, idx, "objs", f"{link}", f"{link}.usd"), f"l_{link}", origins[link])
        else:
            l = links[f"l_{link}"]
        joint_info = robot_tree[link][1]
        parent = robot_tree[link][0]
        if link == root:
            continue
        if parent is None:
            parent = "world"
            pos = origins[link]
            joint_info = {
                "name": get_joint_name("fixed"),
                "type": "fixed",
                "origin": get_translation_matrix(pos[0], pos[1], pos[2])
            }
        if f"l_{parent}" not in links.keys():
            mesh_idx = parent
            if robot_tree.get(parent, None) is not None and robot_tree[parent][1] is not None and robot_tree[parent][1].get("substitute_mesh_idx", None) is not None:
                mesh_idx = robot_tree[parent][1]["substitute_mesh_idx"]
            if mesh_idx != parent:
                shutil.rmtree(os.path.join(path, idx, "objs", f"{parent}"))
                shutil.copytree(os.path.join(path, idx, "objs", f"{mesh_idx}"), os.path.join(path, idx, "objs", f"{parent}"))
                for file in os.listdir(os.path.join(path, idx, "objs", f"{parent}")):
                    if not file.endswith('.mtl') and file.startswith(f"{mesh_idx}"):
                        os.rename(os.path.join(path, idx, "objs", f"{parent}", file), os.path.join(path, idx, "objs", f"{parent}", file.replace(f"{mesh_idx}", f"{parent}")))
            material = None
            if os.path.isfile(os.path.join(path, idx, "objs", f"{mesh_idx}.png")):
                texture = urdfpy.Texture(filename=os.path.join(path, idx, "objs", f"{mesh_idx}.png"))
                material = urdfpy.Material(name=get_link_name("material"), texture=texture)
            if parent != "world":
                collision = None#[urdfpy.Collision(name="temp", origin=None, geometry=urdfpy.Geometry(mesh=urdfpy.Mesh(filename=os.path.join(path, idx, "objs", f"{mesh_idx}",f"{mesh_idx}.obj"))))]
            else:
                collision = None
            p = urdfpy.Link(f'l_{parent}', visuals=[urdfpy.Visual(material=material, geometry=urdfpy.Geometry(mesh=urdfpy.Mesh(filename=os.path.join(path, idx, "objs",f"{mesh_idx}",  f"{mesh_idx}.obj"))))], collisions=collision, inertial=None)
            links[f"l_{parent}"] = p
            #usdutils.add_mesh(os.path.join(path, idx, "objs", f"{parent}", f"{parent}.usd"), f"l_{parent}", origins[parent])
        else:
            p = links[f"l_{parent}"]
        pos_l = origins[link]
        pos_p = origins[parent]
        origin_shift = (pos_l[0] - pos_p[0], pos_l[1] - pos_p[1], pos_l[2] - pos_p[2])
        limit_info = joint_info.get("limit", None)
        # if joint_info.get("type", "fixed") == "prismatic":
        #     if limit_info is not None and limit_info.get("lower", None) is not None:
        #         limit_info["lower"] *= scale_factor
        #         limit_info["upper"] *= scale_factor
        if limit_info:
            limit = urdfpy.JointLimit(limit_info.get("effort", 2000), limit_info.get("velocity", 2), limit_info.get("lower", -1), limit_info.get("upper", 1))
            if limit_info.get("lower_1") or limit_info.get("upper_1"):
                limit_1 = urdfpy.JointLimit(limit_info.get("effort", 2000), limit_info.get("velocity", 2), limit_info.get("lower_1", -1), limit_info.get("upper_1", 1))
        else:
            limit = urdfpy.JointLimit(2000, 2 , -1, 1)
        type = joint_info.get("type", "fixed")
        if type == "fixed" or type == "prismatic":
            shift_axis = joint_info.get("origin_shift", (0, 0, 0))
            j = urdfpy.Joint(get_joint_name(type), joint_info.get("type", "fixed"), f"l_{parent}", f"l_{link}",
                             axis=joint_info.get("axis", None),
                             origin=get_translation_matrix(origin_shift[0] + shift_axis[0], origin_shift[1]+ shift_axis[1], origin_shift[2] + shift_axis[2]),
                             limit=limit)
            joints.append(j)
            #usdutils.add_joint(joint_info.get("name", "temp"), f"l_{parent}", f"l_{link}", type="fixed")
        elif type == "revolute" or type == "continuous":
            shift_axis = joint_info.get("origin_shift", (0, 0, 0))
            l_abstract = urdfpy.Link(f'abstract_{parent}_{link}', visuals=None, collisions=None, inertial=None)
            links[f'abstract_{parent}_{link}'] = l_abstract
            j_real = urdfpy.Joint(get_joint_name(type), type, f"l_{parent}", f'abstract_{parent}_{link}', axis=joint_info.get("axis", None), limit=limit, origin=get_translation_matrix(origin_shift[0] + shift_axis[0], origin_shift[1] + shift_axis[1], origin_shift[2] + shift_axis[2]))
            j_abstract = urdfpy.Joint(get_joint_name("fixed"), "fixed", f"abstract_{parent}_{link}", f"l_{link}", axis=None, limit=None, origin=get_translation_matrix(-shift_axis[0], -shift_axis[1], -shift_axis[2]))
            joints.append(j_real)
            joints.append(j_abstract)
            axis = joint_info.get("axis", None)
            if axis[2] == 1:
                axis = "Z"
            elif axis[1] == 1:
                axis = "Y"
            else:
                axis = "X"
            #usdutils.add_joint(joint_info.get("name", "temp"), f"l_{parent}", f"l_{link}", axis, limit_info.get("lower", -math.inf), limit_info.get("upper", math.inf), type="revolute", shift_from_body0=shift_axis)
        elif type == "revolute_prismatic" or type == "continuous_prismatic":
            shift_axis = joint_info.get("origin_shift", (0, 0, 0))
            l_abstract = urdfpy.Link(get_link_name("abstract"), visuals=None, collisions=None, inertial=None)
            links[l_abstract.name] = l_abstract
            j_real = urdfpy.Joint(joint_info.get("name"), joint_info.get("type").split('_')[0], l_abstract.name, f"l_{link}", axis=joint_info.get("axis"), limit=limit, origin=get_translation_matrix(origin_shift[0] + shift_axis[0], origin_shift[1] + shift_axis[1], origin_shift[2] + shift_axis[2]))
            joints.append(j_real)
            joint_prismatic = urdfpy.Joint(get_joint_name("prismatic"),
                                           "prismatic", f"l_{parent}", l_abstract.name, axis=joint_info.get("axis_1", None), limit=limit_1)
            joints.append(joint_prismatic)
        elif type == "limited_planar":
            all_axis = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
            if not joint_info.get("axis", None) in all_axis:
                print("Invalid axis for limited planar joint")
                raise NotImplementedError
            all_axis.remove(joint_info.get("axis", None))
            fb_lr = all_axis
            abstract_link_1 = urdfpy.Link(get_link_name(f'abstract_planar_primastic_fb'), visuals=None, collisions=None, inertial=None)
            #abstract_link_2 = urdfpy.Link(get_link_name(f'abstract_planar_primastic_lr'), visuals=None, collisions=None, inertial=None)
            abstract_link_3 = urdfpy.Link(get_link_name(f'abstract_planar_revolute'), visuals=None, collisions=None, inertial=None)
            limit_revolute = urdfpy.JointLimit(2000, 2, limit_info.get("lower", -1), limit_info.get("upper", 1))
            limit_prismatic_fb = urdfpy.JointLimit(2000, 2, limit_info.get("lower_1", -1), limit_info.get("upper_1", 1))
            limit_prismatic_lr = urdfpy.JointLimit(2000, 2, limit_info.get("lower_2", -1), limit_info.get("upper_2", 1))
            joint_revolute = urdfpy.Joint(get_joint_name("prismatic_lr"), "prismatic", f"l_{parent}", abstract_link_1.name, axis=fb_lr[1], limit=limit_prismatic_lr, origin=get_translation_matrix(origin_shift[0], origin_shift[1], origin_shift[2]))
            joint_prismatic_fb = urdfpy.Joint(get_joint_name("prismatic_fb"), "prismatic", abstract_link_1.name, abstract_link_3.name, axis=fb_lr[0], limit=limit_prismatic_fb)
            joint_prismatic_lr = urdfpy.Joint(get_joint_name("revolute"), "revolute", abstract_link_3.name, f"l_{link}", axis=joint_info.get("axis", None), limit=limit_revolute)
            joints.append(joint_revolute)
            joints.append(joint_prismatic_fb)
            joints.append(joint_prismatic_lr)
            links[abstract_link_1.name] = abstract_link_1
            #links[abstract_link_2.name] = abstract_link_2
            links[abstract_link_3.name] = abstract_link_3
        elif type == 'flip_revolute':
            limit_flip = urdfpy.JointLimit(2000, 2, limit_info.get("lower", -1), limit_info.get("upper", 1))
            limit_revolute = urdfpy.JointLimit(2000, 2, limit_info.get("lower_1", -1), limit_info.get("upper_1", 1))
            shift_axis_flip = joint_info.get("origin_shift", (0, 0, 0))
            shift_axis_revolute = joint_info.get("origin_shift_1", (0, 0, 0))
            l_abstract_flip = urdfpy.Link(get_link_name("abstract_flip"), visuals=None, collisions=None, inertial=None)
            l_abstract_revolute = urdfpy.Link(get_link_name("abstract_revolute"), visuals=None, collisions=None, inertial=None)
            links[l_abstract_flip.name] = l_abstract_flip
            links[l_abstract_revolute.name] = l_abstract_revolute
            j_flip = urdfpy.Joint(get_joint_name("flip"), "revolute", f"l_{parent}", l_abstract_flip.name, axis=joint_info.get("axis", None), limit=limit_flip, origin=get_translation_matrix(origin_shift[0] + shift_axis_flip[0], origin_shift[1] + shift_axis_flip[1], origin_shift[2] + shift_axis_flip[2]))
            j_revolute = urdfpy.Joint(get_joint_name("revolute"), "revolute", l_abstract_flip.name, l_abstract_revolute.name, axis=joint_info.get("axis_1", None), limit=limit_revolute, origin=get_translation_matrix(shift_axis_revolute[0] - shift_axis_flip[0], shift_axis_revolute[1] - shift_axis_flip[1], shift_axis_revolute[2] - shift_axis_flip[2])) 
            j_fixed = urdfpy.Joint(get_joint_name("fixed"), "fixed", l_abstract_revolute.name, f"l_{link}", axis=None, limit=None, origin=get_translation_matrix(-shift_axis_revolute[0], -shift_axis_revolute[1], -shift_axis_revolute[2]))
            joints.append(j_flip)
            joints.append(j_revolute)
            joints.append(j_fixed)    
        elif type == "limited_floating":
            limit_fb = urdfpy.JointLimit(2000, 2, limit_info.get("lower", -1), limit_info.get("upper", 1))
            limit_lr = urdfpy.JointLimit(2000, 2, limit_info.get("lower_1", -1), limit_info.get("upper_1", 1))
            limit_ud = urdfpy.JointLimit(2000, 2, limit_info.get("lower_2", -1), limit_info.get("upper_2", 1))
            l_abstract_fb = urdfpy.Link(get_link_name("abstract_fb"), visuals=None, collisions=None, inertial=None)
            l_abstract_lr = urdfpy.Link(get_link_name("abstract_lr"), visuals=None, collisions=None, inertial=None)
            j_fb = urdfpy.Joint(get_joint_name("prismatic_fb"), "prismatic", f"l_{parent}", l_abstract_fb.name, axis=(0, 1, 0), limit=limit_fb, origin=None)
            j_lr = urdfpy.Joint(get_joint_name("prismatic_lr"), "prismatic", l_abstract_fb.name, l_abstract_lr.name, axis=(1, 0, 0), limit=limit_lr, origin=None)
            j_ud = urdfpy.Joint(get_joint_name("prismatic"), "prismatic", l_abstract_lr.name, f"l_{link}", axis=(0, 0, 1), limit=limit_ud, origin=get_translation_matrix(origin_shift[0], origin_shift[1], origin_shift[2]))
            joints.append(j_fb)
            joints.append(j_lr)
            joints.append(j_ud)
            links[l_abstract_fb.name] = l_abstract_fb
            links[l_abstract_lr.name] = l_abstract_lr

            


    robot = urdfpy.URDF("scene", list(links.values()), joints=joints)
    robot.save(os.path.join(path, idx, "scene.urdf"))
    shutil.rmtree(os.path.join(path, idx, path, idx, "objs"))
    shutil.copytree(os.path.join(path, idx, "objs"), os.path.join(path, idx, path, idx, "objs"))
    shutil.rmtree(os.path.join(path, idx, "objs"))
    modify_mtl(os.path.join(path, idx))
    if internal_bbox is not None:
        with open(os.path.join(path, idx, "bbox.txt"), 'w') as f:
            for box in internal_bbox:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]}\n")

    #usdutils.save()
    #robot.show()
    robot_tree = {}
    butil.select_none()

def modify_mtl(path):
    for dir in os.listdir(path):
        if str(dir).endswith('.mtl'):
            with open(os.path.join(path, dir), 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.startswith('map_Kd') or line.startswith('map_Pr') or line.startswith('map_Bump'):
                        items = line.split(' ')
                        path_ = items[-1].strip()
                        path_ = path_.split('/')[-1]
                        items[-1] = path_
                        line = ' '.join(items) + '\n'
                        lines[i] = line
            with open(os.path.join(path, dir), 'w+') as f:
                f.writelines(lines)
        elif os.path.isdir(os.path.join(path, dir)):
            modify_mtl(os.path.join(path, dir))



def save_obj_parts_join_objects(
    obj, path=None, idx="unknown", name=None, obj_name=None, first=True
):
    butil.select_none()
    if not isinstance(obj, list):
        obj = [obj]
    if name is None:
        name = "unknown"
    if not isinstance(name, list):
        name = [name] * len(obj)
    if len(obj) == 1:
        return obj[0]
    # The original render engine and world node_tree should be memorized
    # original_render_engine = bpy.context.scene.render.engine
    # original_node_tree = bpy.context.scene.world.node_tree
    # Save a reference to the original scene
    original_scene = bpy.context.scene
    save_parts_export_obj_normalized_json(obj, path, idx, name, obj_name, first)
    # We need to link all these objects into view_layer
    view_layer = bpy.context.view_layer
    for part in obj:
        if part.name not in view_layer.objects:
            view_layer.active_layer_collection.collection.objects.link(part)
    # bpy.context.scene.render.engine = original_render_engine
    # Now switch back to the original scene
    bpy.context.window.scene = original_scene
    bpy.context.view_layer.objects.active = obj[0]
    butil.select_none()
    butil.select(obj)
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.location = 0, 0, 0
    obj.rotation_euler = 0, 0, 0
    obj.scale = 1, 1, 1
    save_whole_object_normalized(obj, path, idx)
    butil.select_none()
    return obj


def save_obj_parts_add(
    obj, path=None, idx="unknown", name=None, obj_name=None, first=True, use_bpy=False, parent_obj_id=None, joint_info=None, material=None, before_export=None
):
    global saved_objs
    butil.select_none()
    if not isinstance(obj, list):
        obj = [obj]
    if name is None:
        name = "unknown"
    if not isinstance(name, list):
        name = [name] * len(obj)
    view_layer = bpy.context.view_layer
    if isinstance(obj, list):
        for o in obj:
            o_ = butil.deep_clone_obj(o, keep_materials=True, keep_modifiers=True)
            #view_layer.active_layer_collection.collection.objects.link(o_)
            saved_objs.append(o_)
    else:
        obj_ = butil.deep_clone_obj(obj, keep_materials=True, keep_modifiers=True)
        #view_layer.active_layer_collection.collection.objects.link(obj_)
        saved_objs.append(obj_)
    # The original render engine and world node_tree should be memorized
    # original_render_engine = bpy.context.scene.render.engine
    # original_node_tree = bpy.context.scene.world.node_tree
    # Save a reference to the original scene
    original_scene = bpy.context.scene
    saved = save_part_export_obj_normalized_add_json(obj, path, idx, name, first=first, use_bpy=use_bpy, parent_obj_id=parent_obj_id, joint_info=joint_info, material=material, before_export=before_export)
    # We need to link all these objects into view_layer
    view_layer = bpy.context.view_layer
    for part in obj:
        if part.name not in view_layer.objects:
            view_layer.active_layer_collection.collection.objects.link(part)
    # bpy.context.scene.render.engine = original_render_engine
    # Now switch back to the original scene
    bpy.context.window.scene = original_scene
    bpy.context.view_layer.objects.active = obj[0]
    butil.select_none()
    return saved


def save_parts_export_obj(
    parts, path=None, idx="unknown", name=None, obj_name=None, first=True
):
    assert len(parts) == len(name)
    if idx == "unknown":
        idx = f"random_{np.random.randint(0, 10000)}"
    else:
        idx = str(idx)
    if not isinstance(obj_name, str):
        obj_name = "unknown"
    if path is None:
        path = os.path.join(os.path.curdir, "outputs")
    if not os.path.exists(path):
        os.makedirs(path)
    if not isinstance(path, Path):
        path = Path(path)
    (path / (idx)).mkdir(exist_ok=True)
    (path / (idx) / "objs").mkdir(exist_ok=True)
    # (path / (idx) / "point_cloud").mkdir(exist_ok=True)
    butil.select_none()

    json_path = os.path.join(path, f"data_infos_{idx}.json")
    if not os.path.exists(json_path):
        infos = []
    else:
        with open(json_path, "r") as f:
            infos = json.load(f)

    if infos and not first:
        info_case = infos[-1]
    else:
        info_case = {}

    info_case["id"] = idx

    info_case["obj_name"] = obj_name
    # info_case["category"] = category
    info_case["file_obj_path"] = os.path.join(path, f"{idx}/objs/whole.obj")
    # info_case["file_pcd_path"] = os.path.join(path, f"{idx}/point_cloud/whole.npz")
    info_parts = info_case.get("part", [])
    length = len(info_parts)
    for i, part in enumerate(parts):
        # Reference the current view layer
        view_layer = bpy.context.view_layer
        # Link the object to the active view layer's collection
        if part.name not in view_layer.objects:
            view_layer.active_layer_collection.collection.objects.link(part)
        bpy.ops.object.select_all(action="DESELECT")

        # Select the current object
        part.select_set(True)
        # Create a new scene
        #new_scene = bpy.data.scenes.new(f"Scene_for_{part.name}")
        # Link the object to the new scene
        #new_scene.collection.objects.link(part)
        # Make the new scene active
        #bpy.context.window.scene = new_scene
        # File path for saving the .blend file
        file_path = os.path.join(path, idx, f"objs/{str(i+length)}.obj")
        # Save the current scene as a new .obj file
        trimesh_object = obj2trimesh(part)
        trimesh_object.export(file_path)

        # write json here
        part_info = {}
        part_info["part_name"] = name[i] + "_part"
        part_info["file_name"] = str(i + length) + ".obj"
        part_info["file_obj_path"] = os.path.join(
            path, f"{idx}/objs/{str(i+length)}.obj"
        )
        # pcd_path = os.path.join(path, f"{idx}/point_cloud/{str(i+length)}")
        # part_info["file_pcd_path"] = pcd_path + ".npz"
        info_parts.append(part_info)

    info_case["part"] = info_parts
    if first:
        infos.append(info_case)
    with open(json_path, "w") as f:
        json.dump(infos, f, indent=2)

    butil.select_none()


def save_parts_export_obj_normalized_json(
    parts, path=None, idx="unknown", name=None, obj_name=None, first=True
):
    assert len(parts) == len(name)
    if idx == "unknown":
        idx = f"random_{np.random.randint(0, 10000)}"
    else:
        idx = str(idx)
    if isinstance(obj_name, str):
        obj_name = str(obj_name)
    if path is None:
        path = os.path.join(os.path.curdir, "outputs")
    if not os.path.exists(path):
        os.makedirs(path)
    if not isinstance(path, Path):
        path = Path(path)
    (path / idx).mkdir(exist_ok=True)
    (path / idx / "objs").mkdir(exist_ok=True)
    # (path / idx / "point_cloud").mkdir(exist_ok=True)
    butil.select_none()

    json_path = os.path.join(path, f"data_infos_{idx}.json")
    if not os.path.exists(json_path):
        infos = []
    else:
        with open(json_path, "r") as f:
            infos = json.load(f)

    if infos and not first:
        info_case = infos[-1]
    else:
        info_case = {}
    info_case["id"] = idx

    info_case["obj_name"] = obj_name
    # info_case["category"] = category
    info_case["file_obj_path"] = os.path.join(path, f"{idx}/objs/whole.obj")
    # info_case["file_pcd_path"] = os.path.join(path, f"{idx}/point_cloud/whole.npz")
    info_parts = info_case.get("part", [])
    length = len(info_parts)
    for i, part in enumerate(parts):
        # Reference the current view layer
        view_layer = bpy.context.view_layer
        # Link the object to the active view layer's collection
        if part.name not in view_layer.objects:
            view_layer.active_layer_collection.collection.objects.link(part)
        bpy.ops.object.select_all(action="DESELECT")
        # Select the current object
        part.select_set(True)
        # Create a new scene
        new_scene = bpy.data.scenes.new(f"Scene_for_{part.name}")
        # Link the object to the new scene
        new_scene.collection.objects.link(part)
        # Make the new scene active
        bpy.context.window.scene = new_scene
        # File path for saving the .blend file
        file_path = os.path.join(path, idx, f"objs/{i+length}.obj")

        # Save the current scene as a new .obj file
        trimesh_object = obj2trimesh(part)
        trimesh_object.export(file_path)

        # write json here
        part_info = {}
        part_info["part_name"] = name[i] + "_part"
        part_info["file_name"] = str(i + length) + ".obj"
        part_info["file_obj_path"] = os.path.join(
            path, f"{idx}/objs/{str(i+length)}.obj"
        )
        # pcd_path = os.path.join(path, f"{idx}/point_cloud/{str(i+length)}")
        # part_info["file_pcd_path"] = pcd_path + ".npz"
        info_parts.append(part_info)

    info_case["part"] = info_parts
    if first:
        infos.append(info_case)
    with open(json_path, "w") as f:
        json.dump(infos, f, indent=2)

    butil.select_none()
saved_obj = 1
def export_curr_scene(
    objs: bpy.types.Object,
    output_folder: Path,
    format="usdc",
    image_res=1024,
    vertex_colors=False,
    individual_export=True,
    before_export=None
) -> Path:
    import time as _time
    _t_export_start = _time.time()
    global saved_obj, saved_objs
    #wait = input("Press Enter to continue.")
    export_usd = format in ["usda", "usdc"]
    #create a new scene
    export_folder = output_folder
    export_folder.mkdir(exist_ok=True)
    export_file = export_folder / output_folder.with_suffix(f".{format}").name

    _t0 = _time.time()
    export.remove_obj_parents()
    #export.delete_objects()
    export.triangulate_meshes()
    _t_prep = _time.time() - _t0
    print(f"[PROFILE] export_curr_scene prep (remove_parents+triangulate): {_t_prep:.1f}s")
    #export.rename_all_meshes()

    scatter_cols = []
    if export_usd:
        if bpy.data.collections.get("scatter"):
            scatter_cols.append(bpy.data.collections["scatter"])
        if bpy.data.collections.get("scatters"):
            scatter_cols.append(bpy.data.collections["scatters"])
        for col in scatter_cols:
            for obj in col.all_objects:
                export.remove_shade_smooth(obj)

    # remove 0 polygon meshes except for scatters
    if export_usd:
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and len(obj.data.polygons) == 0:
                if scatter_cols is not None:
                    if any(x in scatter_cols for x in obj.users_collection):
                         continue
                bpy.data.objects.remove(obj, do_unlink=True)

    collection_views, obj_views = export.update_visibility()

    _t0 = _time.time()
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj not in list(bpy.context.view_layer.objects):
            continue
        if export_usd:
            export.apply_all_modifiers(obj)
        else:
            export.realizeInstances(obj)
            export.apply_all_modifiers(obj)
    _t_modifiers = _time.time() - _t0
    print(f"[PROFILE] export_curr_scene apply_modifiers: {_t_modifiers:.1f}s")

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 1  # choose render sample
    # Set the tile size
    bpy.context.scene.cycles.tile_x = image_res
    bpy.context.scene.cycles.tile_y = image_res

    # iterate through all objects and bake them
    _t0 = _time.time()
    export.bake_scene(
        folderPath=export_folder / "textures",
        image_res=image_res,
        vertex_colors=vertex_colors,
        export_usd=export_usd,
        objs=objs,
    )
    _t_bake = _time.time() - _t0
    print(f"[PROFILE] export_curr_scene bake_scene total: {_t_bake:.1f}s")

    for collection, status in collection_views.items():
        collection.hide_render = status

    for obj, status in obj_views.items():
        obj.hide_render = status

    #export.clean_names()

    for obj in bpy.data.objects:
        obj.hide_viewport = obj.hide_render

    if individual_export:
        _t0 = _time.time()
        for obj in objs:
            butil.select_none()
            export_subfolder = export_folder / obj.name
            export_subfolder.mkdir(exist_ok=True)
            export_file = export_subfolder / f"{obj.name}.{format}"

            obj.hide_viewport = False
            obj.select_set(True)
            if not before_export is None:
                before_export(obj)
                saved_objs.pop()
                saved_objs.append(obj)
            export.run_blender_export(export_file, format, vertex_colors, individual_export)
            saved_obj = obj.copy()
            #bpy.context.scene.objects.active = obj
            #obj.select_set(False)
        _t_obj_export = _time.time() - _t0
        print(f"[PROFILE] export_curr_scene obj_export: {_t_obj_export:.1f}s")
        _t_total_export = _time.time() - _t_export_start
        print(f"[PROFILE] export_curr_scene TOTAL: {_t_total_export:.1f}s")
        #shutil.rmtree(export_folder / "textures")
        return export_file
from infinigen.core import tags as t
def apply(obj, shader_func, selection=None, *args, **kwargs):
    if not isinstance(obj, Iterable):
        obj = [obj]
    if isinstance(shader_func, Callable):
        material = surface.shaderfunc_to_material(shader_func, *args, **kwargs)
    else:
        material = shader_func
    for o in obj:
        index = len(o.data.materials)
        o.data.materials.append(material)
        material_index = read_material_index(o)
        full_like = np.full_like(material_index, index)
        if selection is None:
            material_index = full_like
        elif isinstance(selection, t.Tag):
            sel = tagging.tagged_face_mask(o, selection)
            material_index = np.where(sel, index, material_index)
        elif isinstance(selection, str):
            try:
                sel = surface.read_attr_data(o, selection.lstrip("!"), "FACE")
                material_index = np.where(
                    1 - sel if selection.startswith("!") else sel, index, material_index
                )
            except KeyError:
                material_index = np.zeros(len(material_index), dtype=int)
        else:
            material_index = np.where(selection, index, material_index)
        write_material_index(o, material_index)

def save_part_export_obj_normalized_add_json(
    parts, path=None, idx="unknown", name=None, use_bpy=False, first=True, parent_obj_id=None, joint_info=None, material=None, before_export=None
):
    #render_object_texture_and_save(material, 'res.png')
    global robot_tree, root
    assert len(parts) == len(name)
    if not isinstance(material, list):
        material = [material]
    if idx == "unknown":
        idx = f"random_{np.random.randint(0, 10000)}"
    else:
        idx = str(idx)
    if path is None:
        path = os.path.join(os.path.curdir, "outputs")
    if not os.path.exists(path):
        os.makedirs(path)
    if not isinstance(path, Path):
        path = Path(path)
    (path / idx).mkdir(exist_ok=True)
    (path / idx / "objs").mkdir(exist_ok=True)
    # (path / idx / "point_cloud").mkdir(exist_ok=True)
    butil.select_none()

    json_path = os.path.join(path, f"data_infos_{idx}.json")
    if not os.path.exists(json_path):
        infos = []
        first = True
    else:
        with open(json_path, "r") as f:
            infos = json.load(f)

    if infos and not first:
        info_case = infos[-1]
    else:
        info_case = {}

    # info_parts = info_case["part"]
    info_parts = info_case.get("part", [])
    length = len(info_parts)
    saved = []
    for i, part in enumerate(parts):
        # Reference the current view layer
        view_layer = bpy.context.view_layer
        # if not part.name in bpy.context.collection.objects.keys():
            # bpy.context.collection.objects.link(part)
        # Link the object to the active view layer's collection
        if part.name not in view_layer.objects:
            view_layer.active_layer_collection.collection.objects.link(part)
        # butil.select_none()
        # bpy.context.view_layer.objects.active = part  # Set the object as active
        bpy.ops.object.select_all(action="DESELECT")
        # for m in part.data.materials:
        #     part.data.materials.remove(m)
        # Select the current object
        part.select_set(True)
        for m in material:
            if m is not None:
                if isinstance(m, bpy.types.Material):
                    part.data.materials.append(m)
                elif isinstance(m, list):
                    print(m[0])
                    if hasattr(m[0],'__call__'):
                        apply(part, m[0], m[1])
                        #common.apply(part, m[0], m[1])
                    else:
                        input(f"{m[0]} {m[1]}")
                        m[0].apply(part, selection=m[1])
                elif hasattr(m,'__call__'):
                    surface.add_material(part, m, None)
                    #common.apply(part, m, None)
                else:
                    m.apply(part)
        # Create a new scene
        # new_scene = bpy.data.scenes.new(f"Scene_for_{part.name}")
        # Link the object to the new scene
        # new_scene.collection.objects.link(part)
        # Make the new scene active
        # bpy.context.window.scene = new_scene
        # File path for saving the .blend file
        file_path = os.path.join(path, idx, f"objs/{i + length}.obj")
        #part.data.materials.append(material)
        #render_object_texture_and_save(part, material, os.path.join(path, idx, f"objs/{i + length}.png"))
        if parent_obj_id != "":
            robot_tree[i + length] = [parent_obj_id, joint_info]
            if parent_obj_id is not None and root is None:
                root = i + length
        saved.append(i + length)
        if use_bpy:
            #export.run_blender_export(Path(file_path), 'obj', True, True)
            #bpy.ops.export_scene.obj(filepath=file_path, use_selection=True)
            #os.remove(os.path.join(path, idx, f"objs/{i + length}.mtl"))
            export_curr_scene([part], Path(os.path.join(path, idx, f"objs/{i + length}")), format="obj", image_res=1024, vertex_colors=False, individual_export=True, before_export=before_export)

        # Save the current scene as a new .obj file
        else:
            trimesh_object = obj2trimesh(part)
            trimesh_object.export(file_path)
        part.select_set(False)

        # write json here
        part_info = {}
        part_info["part_name"] = name[i] + "_part"
        part_info["file_name"] = str(i + length) + ".obj"
        part_info["file_obj_path"] = os.path.join(
            path, f"{idx}/objs/{str(i + length)}.obj"
        )
        # pcd_path = os.path.join(path, f"{idx}/point_cloud/{str(i + length)}")
        # part_info["file_pcd_path"] = pcd_path + ".npz"
        info_parts.append(part_info)

    info_case["part"] = info_parts
    if first:
        infos.append(info_case)
        print(infos)
    with open(json_path, "w") as f:
        json.dump(infos, f, indent=2)

    butil.select_none()
    return saved


def save_objects(obj, path=None, idx="unknown", name=None):
    butil.select_none()
    if not isinstance(obj, list):
        obj = [obj]
    if name is not None and not isinstance(name, list):
        name = [name] * len(obj)
    if len(obj) == 1:
        return obj[0]
    # The original render engine and world node_tree should be memorized
    # original_render_engine = bpy.context.scene.render.engine
    # original_node_tree = bpy.context.scene.world.node_tree
    # Save a reference to the original scene
    original_scene = bpy.context.scene
    save_parts(obj, path, idx, name)
    # We need to link all these objects into view_layer
    view_layer = bpy.context.view_layer
    for part in obj:
        if part.name not in view_layer.objects:
            view_layer.active_layer_collection.collection.objects.link(part)
    # bpy.context.scene.render.engine = original_render_engine
    # Now switch back to the original scene
    bpy.context.window.scene = original_scene
    bpy.context.view_layer.objects.active = obj[0]
    butil.select_none()
    # return obj


def save_objects_obj(
    obj, path=None, idx="unknown", name=None, obj_name=None, first=True
):
    butil.select_none()
    if not isinstance(obj, list):
        obj = [obj]
    if name is not None and not isinstance(name, list):
        name = [name] * len(obj)
    # The original render engine and world node_tree should be memorized
    # original_render_engine = bpy.context.scene.render.engine
    # original_node_tree = bpy.context.scene.world.node_tree
    # Save a reference to the original scene
    original_scene = bpy.context.scene
    save_parts_export_obj(obj, path, idx, name, obj_name, first)
    # We need to link all these objects into view_layer
    view_layer = bpy.context.view_layer
    for part in obj:
        if part.name not in view_layer.objects:
            view_layer.active_layer_collection.collection.objects.link(part)
    # bpy.context.scene.render.engine = original_render_engine
    # Now switch back to the original scene
    bpy.context.window.scene = original_scene
    bpy.context.view_layer.objects.active = obj[0]
    butil.select_none()
    # return obj


def save_file_path(path=None, name=None, idx=None, i=999):
    if idx == "unknown":
        idx = f"random_{np.random.randint(0, 10000)}"
    else:
        idx = str(idx)
    if path is None:
        path = os.path.join(os.path.curdir, "outputs")
    if not os.path.exists(path):
        os.mkdir(path)
    if not isinstance(path, Path):
        path = Path(path)
    (path / ("parts_seed_" + idx)).mkdir(exist_ok=True)
    save_path = os.path.join(path, f"parts_seed_{idx}", f"{name}_{i}.blend")
    if os.path.exists(save_path):
        before_path = os.path.join(path, f"parts_seed_{idx}_before")
        if not os.path.exists(before_path):
            os.mkdir(before_path)
        os.rename(
            save_path,
            os.path.join(
                before_path,
                f"{name}_{i}_before_random_{np.random.randint(0, 10000)}.blend",
            ),
        )
    return save_path


def save_file_path_obj(path=None, name=None, idx=None, i=999):
    if idx == "unknown":
        idx = f"random_{np.random.randint(0, 10000)}"
    else:
        idx = str(idx)
    if path is None:
        path = os.path.join(os.path.curdir, "outputs")
    if not os.path.exists(path):
        os.mkdir(path)
    if not isinstance(path, Path):
        path = Path(path)
    (path / idx).mkdir(exist_ok=True)
    save_path = os.path.join(path, f"{idx}", f"{i}.obj")
    if os.path.exists(save_path):
        before_path = os.path.join(path, f"{idx}_before")
        if not os.path.exists(before_path):
            os.mkdir(before_path)
        os.rename(
            save_path,
            os.path.join(
                before_path,
                f"{name}_{i}_before_random_{np.random.randint(0, 10000)}.obj",
            ),
        )
    return save_path


def separate_loose(obj):
    select_none()
    objs = butil.split_object(obj)
    i = np.argmax([len(o.data.vertices) for o in objs])
    obj = objs[i]
    objs.remove(obj)
    butil.delete(objs)
    return obj


def print3d_clean_up(obj):
    bpy.ops.preferences.addon_enable(module="object_print3d_utils")
    with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.mesh.fill_holes()
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.mesh.normals_make_consistent()
        bpy.ops.mesh.print3d_clean_distorted()
        bpy.ops.mesh.print3d_clean_non_manifold()
