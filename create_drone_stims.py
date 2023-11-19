from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
from itertools import combinations, permutations, product


# Set the coordinates
def getCoords():
    """Coordinates for placement of drone features

    Returns:
        dict: dictionary of features and their coordinates
    """
    coords = {
    '0':   {'A-1': (1230, -380),
            'A-2': (1230, -380),
            'A-3': (1230, -380),
            'A-4': (1230, -380),
            'B-1': (1230, -370),
            'B-2': (1230, -370),
            'B-3': (1230, -370),
            'B-4': (1230, -370),
            'C-1': (1230, -350),
            'C-2': (1230, -350),
            'C-3': (1230, -350),
            'C-4': (1230, -350),
            'D-1': (1200, -500),
            'D-2': (1220, -370),
            'D-3': (1180, -345),
            'D-4': (1090, -430),
            'E-1': (1155, -510),
            'E-2': (1155, -530),
            'E-3': (1195, -440),
            'E-4': (1180, -460)},
    '1':   {'A-1': (2220, -100),
            'A-2': (2220, -100),
            'A-3': (2220, -100),
            'A-4': (2220, -100),
            'B-1': (2200, -100),
            'B-2': (2210, -90),
            'B-3': (2210, -90),
            'B-4': (2210, -90),
            'C-1': (2210, -90),
            'C-2': (2210, -90),
            'C-3': (2210, -90),
            'C-4': (2210, -90),
            'D-1': (2120, -100),
            'D-2': (2240, -90),
            'D-3': (2055, -100),
            'D-4': (2115, -100),
            'E-1': (2200, -205),
            'E-2': (2210, -215),
            'E-3': (2210, -150),
            'E-4': (2190, -195)},
    '2':   {'A-1':(2820,730),
            'A-2':(2820,730),
            'A-3':(2820,730),
            'A-4':(2820,730),
            'B-1':(2820,730),
            'B-2':(2820,730),
            'B-3':(2820,730),
            'B-4':(2820,730),
            'C-1':(2820,730),
            'C-2':(2820,730),
            'C-3':(2820,730),
            'C-4':(2820,730),
            'D-1':(2720,620),
            'D-2':(2770,750),
            'D-3':(2650,780),
            'D-4':(2650,750),
            'E-1':(2810,640),
            'E-2':(2820,630),
            'E-3':(2820,700),
            'E-4':(2820,700)},
    '3':   {'A-1':(2820,1780),
            'A-2':(2820,1770),
            'A-3':(2820,1770),
            'A-4':(2820,1770),
            'B-1':(2820,1760),
            'B-2':(2820,1760),
            'B-3':(2820,1760),
            'B-4':(2820,1760),
            'C-1':(2820,1770),
            'C-2':(2820,1770),
            'C-3':(2820,1770),
            'C-4':(2820,1770),
            'D-1':(2720,1640),
            'D-2':(2770,1720),
            'D-3':(2650,1640),
            'D-4':(2650,1640),
            'E-1':(2810,1710),
            'E-2':(2820,1720),
            'E-3':(2820,1740),
            'E-4':(2820,1750)},
    '4':   {'A-1': (2220, 2570),
            'A-2': (2220, 2570),
            'A-3': (2220, 2570),
            'A-4': (2220, 2570),
            'B-1': (2200, 2570),
            'B-2': (2200, 2570),
            'B-3': (2200, 2570),
            'B-4': (2200, 2570),
            'C-1': (2210, 2570),
            'C-2': (2210, 2570),
            'C-3': (2210, 2570),
            'C-4': (2210, 2570),
            'D-1': (2120, 2400),
            'D-2': (2165, 2500),
            'D-3': (2000, 2570),
            'D-4': (2050, 2400),
            'E-1': (2180, 2550),
            'E-2': (2170, 2540),
            'E-3': (2200, 2560),
            'E-4': (2190, 2540)},
    '5':   {'A-1': (1225, 2890),
            'A-2': (1225, 2890),
            'A-3': (1225, 2890),
            'A-4': (1225, 2890),
            'B-1': (1225, 2880),
            'B-2': (1225, 2880),
            'B-3': (1225, 2880),
            'B-4': (1225, 2880),
            'C-1': (1225, 2880),
            'C-2': (1225, 2880),
            'C-3': (1225, 2880),
            'C-4': (1225, 2880),
            'D-1': (1200, 2750),
            'D-2': (1225, 2860),
            'D-3': (980, 2860),
            'D-4': (1080, 2860),
            'E-1': (1155, 2890),
            'E-2': (1150, 2890),
            'E-3': (1195, 2880),
            'E-4': (1175, 2890)},
    '6':   {'A-1': (250, 2590),
            'A-2': (240, 2580),
            'A-3': (240, 2580),
            'A-4': (240, 2580),
            'B-1': (240, 2580),
            'B-2': (240, 2580),
            'B-3': (240, 2580),
            'B-4': (240, 2580),
            'C-1': (270, 2580),
            'C-2': (270, 2580),
            'C-3': (270, 2580),
            'C-4': (270, 2580),
            'D-1': (270, 2400),
            'D-2': (255, 2500),
            'D-3': (105, 2570),
            'D-4': (105, 2400),
            'E-1': (120, 2570),
            'E-2': (120, 2570),
            'E-3': (190, 2570),
            'E-4': (170, 2560)},
    '7':   {'A-1':(-350,1780),
            'A-2':(-360,1770),
            'A-3':(-360,1770),
            'A-4':(-360,1770),
            'B-1':(-350,1760),
            'B-2':(-350,1760),
            'B-3':(-350,1760),
            'B-4':(-350,1760),
            'C-1':(-340,1770),
            'C-2':(-340,1770),
            'C-3':(-340,1770),
            'C-4':(-340,1770),
            'D-1':(-340,1640),
            'D-2':(-340,1720),
            'D-3':(-400,1640),
            'D-4':(-400,1640),
            'E-1':(-500,1710),
            'E-2':(-510,1720),
            'E-3':(-420,1740),
            'E-4':(-430,1750)},
    '8':   {'A-1':(-350,730),
            'A-2':(-360,730),
            'A-3':(-360,730),
            'A-4':(-360,730),
            'B-1':(-350,730),
            'B-2':(-350,730),
            'B-3':(-350,730),
            'B-4':(-350,730),
            'C-1':(-340,730),
            'C-2':(-340,730),
            'C-3':(-340,730),
            'C-4':(-340,730),
            'D-1':(-340,620),
            'D-2':(-340,750),
            'D-3':(-400,780),
            'D-4':(-400,750),
            'E-1':(-500,640),
            'E-2':(-510,630),
            'E-3':(-420,700),
            'E-4':(-430,700)},
    '9':   {'A-1': (270, -100),
            'A-2': (270, -100),
            'A-3': (270, -100),
            'A-4': (270, -100),
            'B-1': (270, -90),
            'B-2': (270, -90),
            'B-3': (270, -90),
            'B-4': (270, -90),
            'C-1': (270, -90),
            'C-2': (270, -90),
            'C-3': (270, -90),
            'C-4': (270, -90),
            'D-1': (270, -100),
            'D-2': (230, -90),
            'D-3': (170, -100),
            'D-4': (70, -100),
            'E-1': (150, -225),
            'E-2': (130, -235),
            'E-3': (195, -150),
            'E-4': (195, -185)}}
    
    return coords


def getPairs():
    """For the reduced set of features, get the pairs

    Returns:
        numpy.array: array of pairs by rows
    """
    pairs = np.array([['A-1','A-2'],
                    ['B-1','B-2'],
                    ['C-1','C-2'],
                    ['D-1','D-2'],
                    ['E-1','E-2'],
                    ['A-3','A-4'],
                    ['B-3','B-4'],
                    ['C-3','C-4'],
                    ['D-3','D-4'],
                    ['E-3','E-4']])
    return pairs


def createDroneIndices(N):
    """Creates indices for the drone features based on number of stimuli

    Args:
        N (int): number of stimuli to be created

    Returns:
        numpy.arrays: feature_indices, coordinate_indices
    """
    feature_indices = np.array((list(product((1,0), repeat=N))))
    coordinate_indices = np.array(list(permutations(np.arange(N)[::2],int(N/2))))
    return feature_indices, coordinate_indices


def makeFeatureCombos(feature_indices, pairs):
    """Indexes the pairs to create the feature combinations

    Args:
        feature_indices (numpy.array): matrix of indices for the features
        pairs (numpy.array): the pairs of features

    Returns:
        list: feature_combos
    """
    feature_combos = []
    for i in range(feature_indices.shape[0]):
        indx = feature_indices[i,:]
        feature_combos.append(pairs[np.arange(len(indx)), indx])
    return feature_combos


def createDrone(feature_combo,features,image_scale=1.4,center=(550,550),places=None,component_dir=None,
                save_dir='drone_pngs/',save_bool=False,coords=None):
    """Creates an individaul drone image

    Args:
        feature_combo (list): combination of features at each location
        features (list): file names of the features
        image_scale (float, optional): size of image relative to the component sizes. Defaults to 1.4.
        center (tuple, optional): where to center the image. Defaults to (550,550).
        places (list of str, optional): Spoke features are located upon. Defaults to None.
        component_dir (str, optional): file directory for feature components. Defaults to None.
        save_dir (str, optional): file directory for where to save the stimulus. Defaults to 'drone_pngs/'.
        save_bool (bool, optional): whether to save the file. Defaults to False.
        coords (dict, optional): coordinates for each feature type in each location. Defaults to None.

    Returns:
        drone_img, drone_name
    """
    if places is None:
        places = ['0','1','2','3','4','5','6','7','8','9']
    if component_dir is None:
        component_dir = '/drone_components/png_components'
    if coords is None:
        coords = getCoords()
        
    # Load the images
    feature_imgs = {}
    for feature in features:
        feature_imgs[feature] = Image.open(os.path.join(component_dir,feature+'.png')).convert('RGBA')
    # # Open the images
    drone_frame_img = Image.open(os.path.join(component_dir,'drone_frame.png')).convert('RGBA')

    # Create a new image with the same size as the drone frame, but scaled
    drone_img = Image.new('RGBA', tuple(np.round(np.array(drone_frame_img.size)*image_scale).astype(int)), (0, 0, 0, 0))

    # Paste drone onto the new image at the center
    drone_img.paste(drone_frame_img, center,drone_frame_img)

    # Place each of the features
    for place in places:
        coord_base = coords[place][feature_combo[int(place)]]
        coord = shiftCoord(coord_base, center)
        img = feature_imgs[feature_combo[int(place)]]
        drone_img.paste(img, tuple(coord),img)
    
    # Create the descriptive name
    drone_name = '_'.join([f'{coord}-{feature}' for coord,  feature in zip(np.arange(len(feature_combo)),feature_combo)])
    
    # Save it
    if save_bool:
        drone_img.save(os.path.join(save_dir,drone_name+'.png'))
        
    return drone_img, drone_name


def makeRandomAB():
    """Randomly generates A and B prototypes with opposite feature combinations

    Returns:
        _type_: A_prototype_code, B_prototype_code, A_prototype_name, B_prototype_name
    """
    pairs = np.array([['A-1','A-2'],
                        ['A-3','A-4'],
                        ['B-1','B-2'],
                        ['B-3','B-4'],
                        ['C-1','C-2'],
                        ['C-3','C-4'],
                        ['D-1','D-2'],
                        ['D-3','D-4'],
                        ['E-1','E-2'],
                        ['E-3','E-4']])
    prototype_code = pairs.copy()
    np.random.shuffle(prototype_code)
    # print(prototype_code)
    for i in range(prototype_code.shape[0]):
        prototype_code[i,:] = prototype_code[i,np.random.choice([0,1],2,replace=False)]
    A_prototype_code = prototype_code[:,0]
    B_prototype_code = prototype_code[:,1]

    A_prototype_name = '_'.join([str(num) + '-' + loc for num, loc in zip(range(len(A_prototype_code)),A_prototype_code)])
    B_prototype_name = '_'.join([str(num) + '-' + loc for num, loc in zip(range(len(B_prototype_code)),B_prototype_code)])
    return A_prototype_code, B_prototype_code, A_prototype_name, B_prototype_name


def createDistanceIndices(prototype_code,distance=1,n_stims=10):
    """Randomly selects indices to change based on the feature "distance" from the prototype

    Args:
        prototype_code (numpy.array): features used in the prototoype
        distance (int, optional): the distance from the prototype. Defaults to 1.
        n_stims (int, optional): number of stimuli at that distance to create. Defaults to 10.

    Returns:
        _type_: _description_
    """
    combos = np.array(list(combinations(range(prototype_code.shape[0]),distance)))
    distance_indices = combos[np.random.choice(np.arange(combos.shape[0]),n_stims,replace=False)]
    return distance_indices


def shiftCoord(coord, center):
    """Moves the coordinates relative to the specified center.

    Args:
        coord (list/tuple): coordinates of a feature
        center (list/tuple): coordinates of the center

    Returns:
        tuple: the coordinates shifted relative to the center
    """
    coord = np.array(coord)
    coord[0], coord[1] = coord[0] + center[0], coord[1] + center[1]
    return tuple(coord)