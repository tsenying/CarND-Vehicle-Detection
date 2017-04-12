from skimage.feature import hog

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
    # 414 : white sedan right
    # ind = 414
    print("cars index = {}".format(ind))
    
    # Read in the car image
    car_image = mpimg.imread(cars[ind])
    # Read in the not-car image
    notcar_image = mpimg.imread(notcars[ind])
    
    # Plot the result
    fig, axes = plt.subplots(4, 2, squeeze=False, figsize=(8, 6.5))
    
    # Plot car
    # axes[0][0].imshow(cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB))
    axes[0][0].imshow(car_image)
    axes[0][0].set_title('Car', fontsize=10)
    
    # Plot not car
    # axes[0][1].imshow(cv2.cvtColor(notcar_image, cv2.COLOR_BGR2RGB))
    axes[0][1].imshow(notcar_image)
    axes[0][1].set_title('Not Car channel ', fontsize=10)
    
    channel_index = 0
    color_space = 'RGB'
    car_image = convert_color(car_image, color_space)
    notcar_image = convert_color(notcar_image, color_space) 
    
    
    for channel_index in [0,1,2]:
        #### car ####
        channel_image = car_image[:,:,channel_index]
        print("channel_image={}".format(channel_image))
    
        # Plot car
        axes[channel_index+1][0].imshow(channel_image, cmap='gray')
        axes[channel_index+1][0].set_title('Car channel ' + str(channel_index), fontsize=10)
    
        #### not car ####                    
        channel_image = notcar_image[:,:,channel_index]

        # Plot not car
        axes[channel_index+1][1].imshow(channel_image, cmap='gray')
        axes[channel_index+1][1].set_title('Not Car channel ', fontsize=10)

    plt.show()
