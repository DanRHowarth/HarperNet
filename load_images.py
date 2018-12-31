# required import statements 
import os
import glob                          # library for loading images from a directory
import matplotlib.image as mpimg
from PIL import Image


def image_load(image_dir, folder_names):
    '''
    takes as input a filepath and folders and returns a list of (numpy array, folder, label) items
    image_dir: a filepath to image folders
    folder names: image folder names to be appended
    '''
    
    # create an empty image list to then populate 
    image_list = []
    
    for i,folder in enumerate(folder_names):
    
        # go through all the files in each folder 
        for file in glob.glob(os.path.join(image_dir,folder, "*")):

            # Read in the image - this loads the imges as an array  ## here we could read it in as a PIL format
            image_array = mpimg.imread(file)

            # Check if the image exists/if it's been correctly read-in
            if not image_array is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                image_list.append((image_array, folder, i))
    
    return image_list


def PIL_image_load(image_dir, folder_names):
    '''
    takes as input a filepath and folders and returns a list of (PIL image, folder, label) items
    image_dir: a filepath to image folders
    folder names: image folder names to be appended
    '''
    
    # create an empty image list to then populate 
    image_list = []
    
    for i,folder in enumerate(folder_names):
    
        # go through all the files in each folder 
        for file in glob.glob(os.path.join(image_dir,folder, "*")):

            # Read in the image - this loads the imges as an array  ## here we could read it in as a PIL format
            image_PIL = Image.open(file)

            # Check if the image exists/if it's been correctly read-in
            if not image_PIL is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                image_list.append((image_PIL, folder, i))
    
    return image_list


def category_number_mapping(category_names):    
    '''
    pass in a list of category names
    returns the category - number mapping used when loading images
    '''

    # create an empty list 
    names_to_number = []

    # iterate through the folders
    for i, category in enumerate(category_names):

        # append to our new list in category:number mapping
        names_to_number.append((category, i))

    # return list 
    return names_to_number
