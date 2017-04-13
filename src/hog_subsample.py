import numpy as np
import cv2
from feature_extraction_utils import *
import matplotlib.pyplot as plt
import config

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
    xstart=None, xstop=None):
    
    draw_img = np.copy(img)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #img = img.astype(np.float32)/255
    
    # crop image to region of interest
    if (xstart == None):
        xstart = 0
    if (xstop  == None):
        print("img.shape={}".format(img.shape))
        xstop  = img.shape[1]
    img_tosearch = img[ystart:ystop, xstart:xstop,:]
    
    if ( config.debug ):
        print("find_cars frame={}, xstart={}, xstop={}, img_tosearch.shape={}".format(config.count, xstart, xstop, img_tosearch.shape))
        plt.imshow(img_tosearch)
        plt.show()
    
    # convert image to target color space
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    
    # resize image according to scale
    # if scale is > 1, then resize image smaller making relative size of fixed size search window larger
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    boxes = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
             ypos = yb*cells_per_step
             xpos = xb*cells_per_step
             # Extract HOG for this patch
             hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
             hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
             hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
             hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

             xleft = xpos*pix_per_cell
             ytop = ypos*pix_per_cell

             # Extract the image patch
             subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

             # Get color features
             spatial_features = bin_spatial(subimg, size=spatial_size)
             hist_features = color_hist(subimg, nbins=hist_bins)

             # Scale features 
             test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
             #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
             
             # Compute prediction using classifier
             # prediction = svc.predict(test_features)
             score_threshold = 0.2
             score = svc.decision_function(test_features)
             prediction = int(score > score_threshold)

             if prediction == 1:
                 xbox_left = np.int(xleft*scale)
                 ytop_draw = np.int(ytop*scale)
                 win_draw = np.int(window*scale)
                 box = [ (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart) ]
                 boxes.append(box)
                 cv2.rectangle(draw_img,box[0],box[1],(0,0,255),6) 

    return draw_img, boxes

if __name__ == "__main__":
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import pickle
    import sys
    
    scale = 2.0
    if ( len(sys.argv) > 1 ):
        scale = float( sys.argv[1] )
    print("scale={}".format(scale))
    
    xstart = None
    xstop  = None
    if ( len(sys.argv) > 4 ):
        xstart = float( sys.argv[2] )
        xstop  = float( sys.argv[3] )
        print( "xstart={},xstop={}".format(xstart,xstop) )
        
    dist_pickle = pickle.load( open("svc_trained.p", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    print("svc={},X_scaler={},orient={},pix_per_cell={},cell_per_block={},spatial_size={},hist_bins={}".format(svc,X_scaler,orient,pix_per_cell,cell_per_block,spatial_size,hist_bins))
    
    ystart = 400
    ystop = 656
    
    # Plot the result
    fig, axes = plt.subplots(2, 3, squeeze=False, figsize=(8.5, 4.5))
    # fig.tight_layout()
    plt.suptitle( "Sliding Window: scale={0}, ystart={1}, ystop={2}".format(scale, ystart, ystop, xstart, xstop) )
    

    for i in range(1,7):
        print("processing image " + str(i))
        
        img_path = 'test_images/test{0}.jpg'.format(str(i))
        img = mpimg.imread(img_path)
        
        out_img, boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, xstart, xstop)
    
        print ("boxes count={}, boxes={}".format(len(boxes), boxes ))
        
        row = (i-1)//3
        col = (i-1)%3
        print("i={0},row={1},col={2}".format(i,row,col))
        axes[row][col].imshow(out_img)
        axes[row][col].set_title('Image {0} detections={1}'.format(i, len(boxes)), fontsize=10)
        
    save_plot_path = "output_images/sliding_window_scale_{0}.png".format(scale)
    plt.savefig(save_plot_path)
    plt.show()
