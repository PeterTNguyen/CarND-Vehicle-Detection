#Imports
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label as apply_label
from VehicleDetectionFunctions import *
import pickle

class VehicleDetector:
    
    #Heatmap
    M = 720
    N = 1280
    num_avgs = 10
    scales = [1.0, 1.25, 1.5, 1.75, 2.0, 3.0]
#     scales = [1.0, 1.5, 2.0, 2.5]
    ystart = 400
    ystop = [496, 520, 544, 568, 592, 688]
#     ystop = [496, 544, 592, 640]
    threshold = 6
    
    #Heatmap history
    heatmap = np.zeros((num_avgs, 720, 1280))
    avg_heatmap = np.zeros((720, 1280))
    heatmap_thresh = np.zeros((720, 1280))
    heatmap_idx = 0
    draw_img = np.zeros((720, 1280,3))

    
    #Feature Variables
    hist_bins = 64
    spatial_size=(32,32)
    
    #Hog subsampler Variables
    orient=9
    pix_per_cell=8
    cell_per_block=2
    x_offset = 640

    #Classifier
    X_scaler = None
    svc = LinearSVC()
    
    #HeatMap
    def __init__(self):
        self.svc, self.X_scaler = pickle.load( open('svc.p', 'rb') )
        
        
            # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scale, bool_drawboxes=False):
        img_shape = img.shape
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), 
                                                           np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient*self.cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, 
                                self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, 
                                self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, 
                                self.cell_per_block, feature_vec=False)

        bin_img = np.zeros((img_shape[0], img_shape[1]))
        x_offset_steps = self.x_offset//(64*scale/4)
        for xb in range(nxsteps):
            if xb >= x_offset_steps:
                for yb in range(nysteps):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step
                    # Extract HOG for this patch
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, 
                                     xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, 
                                     xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, 
                                     xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                    xleft = xpos*self.pix_per_cell
                    ytop = ypos*self.pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, 
                                                        xleft:xleft+window], (window,window))
                    spatial_features = bin_spatial(subimg, 'YCrCb', size=self.spatial_size)
                    hist_features = color_hist(subimg, nbins=self.hist_bins)

                    # Scale features and make a prediction
                    test_features = self.X_scaler.transform(np.hstack( 
                        (spatial_features,hist_features,hog_features)).reshape(1, -1))    

                    test_prediction = self.svc.predict(test_features)

                    if (test_prediction == 1) or bool_drawboxes:
                        xbox_left = np.int(xleft*scale)
                        ytop_draw = np.int(ytop*scale)
                        win_draw = np.int(window*scale)
                        ones_mask = np.ones((win_draw,win_draw))
                        bin_img[ytop_draw+ystart:ytop_draw+win_draw+ystart,
                                xbox_left:xbox_left+win_draw] += ones_mask
                        cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),
                                      (xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),3) 

        return draw_img, bin_img
    
    def update_heatmap(self, img, bool_drawboxes=False):
        img_shape = img.shape
        bin_total = np.zeros((img_shape[0], img_shape[1]))
        draw_img = img
        for scale,ystop in zip(self.scales, self.ystop):
            draw_img, bin_img = self.find_cars(draw_img, self.ystart, ystop, scale,
                                              bool_drawboxes=bool_drawboxes)
            bin_total += bin_img
            
        self.draw_img = draw_img
        
        #Update Heatmap
        self.heatmap[self.heatmap_idx,:,:] = bin_total
        
        #Calculate Heatmap average
        self.calc_avg_heatmap()
        
        #Update heatmap idx
        self.heatmap_idx = (self.heatmap_idx+1)%self.num_avgs
        

        
        #Threshold Heatmap
        self.heatmap_thresh = apply_threshold(self.avg_heatmap, self.threshold)
        labels = apply_label(self.heatmap_thresh)
        
        drawn_image = draw_labeled_bboxes(img, labels)
        
        return drawn_image

    def calc_avg_heatmap(self):
        avg_heatmap = np.zeros((720, 1280))
        for i in range(self.num_avgs):
            idx = (self.heatmap_idx + i ) % self.num_avgs
            exp_coeff = math.exp(-1*i/self.num_avgs)
#             print(exp_coeff)
            avg_heatmap += (self.heatmap[idx] * exp_coeff)

            
        self.avg_heatmap = avg_heatmap
            
            
            
            
            
            
            
        