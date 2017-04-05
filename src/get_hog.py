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

if __name__ == "__main__":
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import glob
    import os
    
    # Read in our vehicles and non-vehicles
    #images = glob.glob('*.jpeg')
    images = glob.glob('./images/**/*.jpeg', recursive=True)
    #print("images={}".format(images))
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
    ind = np.random.randint(0, len(cars))
    # Read in the image
    image = mpimg.imread(cars[ind])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient, 
                            pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=False)

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')

    plt.show()
