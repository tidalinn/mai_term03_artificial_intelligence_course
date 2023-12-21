'''Создание модели, которая на вход будет принимать векторы скрытого представления \
цвета и формы лица, параметры камеры и освещения в сцене, а на выходе будет \
генерировать отрендеренное изображение
'''

import pyredner
import torch

def create_model(cam_pos, 
          cam_look_at, 
          shape_coeffs, 
          color_coeffs, 
          ambient_color, 
          dir_light_intensity,
          shape_mean, 
          shape_basis, 
          triangle_list,
          color_mean, 
          color_basis):
    
    indices = triangle_list.permute(1, 0).contiguous() # хранит геометрию усредненного лица в форме полигональной модели
    
    vertices = (shape_mean + shape_basis @ shape_coeffs).view(-1, 3)
    normals = pyredner.compute_vertex_normal(vertices, indices)
    colors = (color_mean + color_basis @ color_coeffs).view(-1, 3)

    m = pyredner.Material(use_vertex_color = True)

    obj = pyredner.Object(
        vertices=vertices, 
        indices=indices, 
        normals=normals, 
        material=m, 
        colors=colors
    )
    
    cam = pyredner.Camera(
        position=cam_pos,
        # Center of the vertices                          
        look_at=cam_look_at,
        up=torch.tensor([0.0, 1.0, 0.0]),
        fov=torch.tensor([45.0]),
        resolution=(256, 256)
    )
    
    scene = pyredner.Scene(camera=cam, objects=[obj])

    ambient_light = pyredner.AmbientLight(ambient_color)
    dir_light = pyredner.DirectionalLight(torch.tensor([0.0, 0.0, -1.0]), dir_light_intensity)

    img = pyredner.render_deferred(scene=scene, lights=[ambient_light, dir_light])
                                   
    return img