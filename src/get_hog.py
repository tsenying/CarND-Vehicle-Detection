from skimage.feature import hog
        
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, 
            pixels_per_cell=(pix_per_cell, pix_per_cell), 
            cells_per_block=(cell_per_block, cell_per_block), 
            visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features= hog(img, orientations=orient, 
            pixels_per_cell=(pix_per_cell, pix_per_cell), 
            cells_per_block=(cell_per_block, cell_per_block), 
            visualise=False, feature_vector=feature_vec)
        return features

def convert_color(img, conv='YCrCb'):
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img
                
if __name__ == "__main__":
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import glob
    import os
    
    # Read in our vehicles and non-vehicles
    #images = glob.glob('*.jpeg')
    images = glob.glob('./images_smallset/**/*.jpeg', recursive=True)
    cars = []
    notcars = []

    #e.g. './images/vehicles_smallset/cars3/998.jpeg'
    for image in images:
        basename = os.path.basename( image )
        if 'image' in basename or 'extra' in basename:
            notcars.append(image)
        else:
            cars.append(image)
    print("len(cars)={}".format(len(cars)))
    print("len(notcars)={}".format(len(notcars)))
        
    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(notcars))
    # 414 : white sedan right # 63 : blue sedan back # 861: red sedan back
    ind = 63
    notcar_ind = 414
    print("cars index = {}".format(ind))
    
    # Read in the car image
    car_image = mpimg.imread(cars[ind])
    # Read in the not-car image
    notcar_image = mpimg.imread(notcars[notcar_ind])
    
    channel_index = 0
    
    # Define HOG parameters
    orient = 12
    pix_per_cell = 16
    cell_per_block = 2
    color_space = 'YCrCb' #'YCrCb' 
    
    
    
    # Plot the result
    fig, axes = plt.subplots(4, 4, squeeze=False, figsize=(8.5, 9.5))
    # fig.tight_layout()
    plt.suptitle('HOG ' + str(color_space) + ' orient=' + str(orient) + ' pix_per_cell=' + str(pix_per_cell) + ' cell_per_block=' + str(cell_per_block))
    
    # Plot car
    axes[0][0].imshow(car_image)
    axes[0][0].set_title('Car', fontsize=10)
    
    axes[0][1].axis('off')
    
    # Plot not car
    axes[0][2].imshow(notcar_image)
    axes[0][2].set_title('Not Car', fontsize=10)
    
    axes[0][3].axis('off')
    
    
    
    car_image = convert_color(car_image, color_space)
    notcar_image = convert_color(notcar_image, color_space) 
    
    print("HOG params orient={}, pix_per_cell={}, cell_per_block={}".format(orient, pix_per_cell, cell_per_block))
    
    for channel_index in [0,1,2]:
        plot_index = channel_index+1
        hog_title = 'HOG channel ' + str(channel_index)
        
        #### car ####
        channel_image = car_image[:,:,channel_index]
        
        # Call our function with vis=True to see an image output
        features, hog_image = get_hog_features(channel_image, orient, 
                                pix_per_cell, cell_per_block, 
                                vis=True, feature_vec=False)
            
        # Plot car
        axes[plot_index][0].imshow(channel_image, cmap='gray')
        axes[plot_index][0].set_title('Car channel ' + str(channel_index), fontsize=10)
        
        axes[plot_index][1].imshow(hog_image, cmap='gray')
        axes[plot_index][1].set_title(hog_title, fontsize=10)
    
        #### not car ####                    
        channel_image = notcar_image[:,:,channel_index]

        # Call our function with vis=True to see an image output
        features, hog_image = get_hog_features(channel_image, orient, 
                                pix_per_cell, cell_per_block, 
                                vis=True, feature_vec=False)

        # Plot not car
        axes[plot_index][2].imshow(channel_image, cmap='gray')
        axes[plot_index][2].set_title('Not Car channel ' + str(channel_index), fontsize=10)

        axes[plot_index][3].imshow(hog_image, cmap='gray')
        axes[plot_index][3].set_title(hog_title, fontsize=10)

    save_plot_path = "output_images/" + "hog_" + str(color_space) + '_' + str(orient) + '_' + str(pix_per_cell) + '_' + str(cell_per_block) + '.png'
    plt.savefig(save_plot_path)
    plt.show()
