#importing some useful packages
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from lib_functions import *
from CameraCalib import *
import math
import numpy as np
import cv2
import pandas as pd
mtx, dist = Cam_Calib()
#src = np.float32([[330,660],[630,430],[650,430],[1060,660]])
#dst = np.float32([[330,660],[330,430],[1060,430],[1060,660]])
thresh_a =0.02
thresh_b =0.2
thresh_c =40
xm_per_pix = 3.7/700 # meters per pixel in x dimension
ym_per_pix = 30/720
src = np.float32([[568, 468], [715, 468], [1040, 680], [270, 680]])
dst = np.float32([[200, 0], [1000, 0], [1000, 680], [200, 680]])
M = cv2.getPerspectiveTransform(src, dst)
Minv=cv2.getPerspectiveTransform(dst, src)
#print(M)
right_fit_coef_arr = np.array([0,0,0], dtype='float') 
left_fit_coef_arr = np.array([0,0,0], dtype='float') 
left_fit_coefdiff_arr = np.array([0,0,0], dtype='float') 
right_fit_coefdiff_arr = np.array([0,0,0], dtype='float') 
clip1 = VideoFileClip("project_video.mp4")
output_video="project_video_out.mp4"
def check_diff(cof_diff):
    if (cof_diff[0] > thresh_a and cof_diff[1] > thresh_b and cof_diff[2] > thresh_c):
        return False
    else:
        return True
def sanity_check(left_x, rightx):
    #diff_fit=rightfit-leftfit
    diff_fitx=rightx-left_x
    diff_fitx*=xm_per_pix
    diff_fitx_avg=np.average(diff_fitx, axis=0)
    diff_fitx_std=np.std(diff_fitx, axis=0)
    #print(diff_fitx_avg)
    #print(diff_fitx_std)
    if diff_fitx_avg <4.7 and diff_fitx_avg >3 and diff_fitx_std <0.5:
        return True
    else:
        return False
def measure_curvature_real(leftx,rightx,ploty):
    ploty*=ym_per_pix
    #leftx_m=np.array(leftx,dtype='float')
    leftx_m=leftx*xm_per_pix
    #rightx_m=np.array(rightx,dtype='float')
    rightx_m=rightx*xm_per_pix
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty, leftx_m, 2)
    right_fit_cr = np.polyfit(ploty, rightx_m, 2)
    left_curverad = pow(1+pow(2*left_fit_cr[0]*y_eval+left_fit_cr[1],2),3/2)/(2*abs(left_fit_cr[0])) ## Implement the calculation of the left line here
    right_curverad = pow(1+pow(2*right_fit_cr[0]*y_eval+right_fit_cr[1],2),3/2)/(2*abs(right_fit_cr[0]))  ## Implement the calculation of the right line here
    return left_curverad, right_curverad
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        # x values of the last n fits of the line
        self.current_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #ploty
        self.ploty = []
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #values of the last n polynomian fits of the line
        self.recent_fits = [] 
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 
def undist_grad_warp(image):
    img_size=(image.shape[1],image.shape[0])
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    color, combined=grad_thresh(undist)
    #M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)
    return undist,warped
left_line = Line()
right_line = Line()
def process_image(image):
    undist_img,warped_img = undist_grad_warp(image)
    """""
    if left_line.detected== False and right_line.detected== False:
            left_fit, right_fit, left_fitx,right_fitx, ploty = fit_polynomial(warped_img)
            left_line.detected= True
            right_line.detected= True
    else:
        left_fit, right_fit,left_fitx,right_fitx, ploty=search_around_poly(warped_img,left_line.best_fit,right_line.best_fit)
    
        if left_line.detected== True and right_line.detected== True:
                left_line.diffs=left_line.recent_fits[len(left_line.recent_fits)-1] -left_fit
                right_line.diffs=right_line.recent_fits[len(right_line.recent_fits)-1] -right_fit
                if check_diff(left_line.diffs)==False or check_diff(right_line.diffs)==False:
                    left_fit, right_fit, left_fitx,right_fitx, ploty = fit_polynomial(warped_img)   
    """""
    left_fit, right_fit, left_fitx,right_fitx, ploty = fit_polynomial(warped_img)
    if sanity_check(left_fitx,right_fitx):
        #left_fit, right_fit, left_fitx,right_fitx, ploty = fit_polynomial(warped_img)
        #left_line.recent_xfitted.append([left_fitx])
        #right_line.recent_xfitted.append([right_fitx])
        left_line.recent_xfitted=left_fitx
        right_line.recent_xfitted=right_fitx
        left_line.current_xfitted=left_fitx
        right_line.current_xfitted=right_fitx
        left_line.bestx=np.average(left_line.recent_xfitted, axis=0)
        right_line.bestx=np.average(right_line.recent_xfitted, axis=0)
        left_line.current_fit=left_fit
        #right_fit_coef_arr=np.vstack((right_fit_coef_arr,right_fit))
        #left_fit_coef_arr=np.vstack((left_fit_coef_arr,left_fit))
        right_line.current_fit=right_fit
        left_line.recent_fits.append(left_line.current_fit)
        right_line.recent_fits.append(right_line.current_fit)
        left_line.best_fit=np.average(left_line.recent_fits, axis=0)
        right_line.best_fit=np.average(right_line.recent_fits, axis=0)
        #left_line.diffs=left_line.recent_fits[len(left_line.recent_fits)-1] -left_line.current_fit
        #right_line.diffs=right_line.recent_fits[len(right_line.recent_fits)-1] -right_line.current_fit
        #left_fit_coefdiff_arr=np.vstack((left_fit_coefdiff_arr,left_fit))
        #right_fit_coefdiff_arr=np.vstack((right_fit_coef_arr,right_fit))
        #processed_image=plot_final(warped_img,left_line.bestx,right_line.bestx,ploty,undist_img)
 
        #processed_image=plot_final(warped_img,left_line.current_xfitted,right_line.current_xfitted,ploty,undist_img)
    else:
        left_line.current_xfitted=left_line.recent_xfitted
        right_line.current_xfitted=right_line.recent_xfitted
    processed_image=plot_final(warped_img,left_line.current_xfitted,right_line.current_xfitted,ploty,undist_img)
    left_Rad, right_rad = measure_curvature_real(left_line.current_xfitted,right_line.current_xfitted,ploty)
    average_curve_rad = (left_Rad + right_rad)/2
    radius_text = "Radius: {0:.2f}m".format(average_curve_rad)
    lane_center = (left_line.current_xfitted[-1] + right_line.current_xfitted[-1])//2
    car_center = processed_image.shape[1]/2  # we assume the camera is centered in the car
    center_offset = (lane_center - car_center) * xm_per_pix
    offset_text="Offset to center: {0:.2f}m".format(center_offset)
    ## area_img writing
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(processed_image, str(radius_text), (60,60), font, 1.25, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(processed_image, str(offset_text), (60,120), font, 1.25, (255,255,255), 2, cv2.LINE_AA)
    #print (left_Rad)
    #print(right_rad)
    

    return processed_image
def plot_final(warped_orig,left_fitx,right_fitx,ploty,undist):
    warp_zero = np.zeros_like(warped_orig).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped_orig.shape[1], warped_orig.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def main():

    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(output_video, audio=False)
if __name__ == "__main__":
    main()