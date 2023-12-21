'''Обработка целевого изображения
'''

import pyredner
import urllib

def prepare_target(path, web = True):
    if web:
        file = 'target.png'
        urllib.request.urlretrieve(path, file)
    else:
        file = path
        
    target = pyredner.imread(file).to(pyredner.get_device())

    return target