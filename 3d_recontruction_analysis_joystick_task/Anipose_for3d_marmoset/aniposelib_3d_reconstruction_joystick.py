import numpy as np
import aniposelib
import toml
import pandas as pd
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames
import os


## caution!! Don't forget to fix the animal swapping problem first!!
# the scripts are in /home/ws523/marmoset_tracking_DLCv2/following_up_analysis/slurm_run_job/marmosets_tracking_3d_training_fixAnimalSwapping_multi_sessions.py


## section 1 - cammera calibration
# calibrate the cameras for 3d

do_cali = 0

# vidnames = [['camera-1.mp4'],
#             ['camera-2.mp4']]
vidnames = [['camera-1_old.mp4'],
            ['camera-2_old.mp4']]

cam_names = ['1', '2']

n_cams = len(vidnames)

#board = CharucoBoard(9, 7,
#                      square_length=22.5, # here, in mm but any unit works
#                      marker_length=16.5,
#                      manually_verify=False)

board = Checkerboard(8, 6,
                      square_length=1, # here, in mm but any unit works
                      manually_verify=False)

# the videos provided are fisheye, so we need the fisheye option
cgroup = CameraGroup.from_names(cam_names, fisheye=False)


if do_cali: 
    # this will take about 15 minutes (mostly due to detection)
    # it will detect the charuco board in the videos,
    # then calibrate the cameras based on the detections, using iterative bundle adjustment
    cgroup.calibrate_videos(vidnames, board)

    # if you need to save and load
    # example saving and loading for later
    cgroup.dump('calibration.toml')

else:
    ## example of loading calibration from a file
    ## you can also load the provided file if you don't want to wait 15 minutes
    cgroup = CameraGroup.load('calibration.toml')


## section 2 - test on videos

# dates for Koala and Vermelho
if 1:
    analyzed_dates = [ 
                      "20231102","20231106","20231107","20231109",
                     ]
    session_start_times = [ 
                           20.5, 14.5, 33.2, 21.0,
                          ] # in second 

    animal1 = "Koala"
    animal2 = "Vermelho"


# session_start_times = [1.00]
# analyzed_dates = ["20221128"]

# get this information using DLC animal tracking GUI, the results are stored: 
# /home/ws523/marmoset_tracking_DLCv2/joystick-weikang-2023-10-31/labeled-data/
lever_locs_all = {'camera-1':{('dodson'):np.array([980, 530]),('scorch'):np.array([980, 530])},
                  'camera-2':{('dodson'):np.array([910,560]),('scorch'):np.array([910, 560])}}

tube_locs_all = {'camera-1':{('dodson'):np.array([750,315]),('scorch'):np.array([1200, 280])},
                 'camera-2':{('dodson'):np.array([1210,700]),('scorch'):np.array([620, 700])}}

boxCorner1_locs_all = {'camera-1':{('dodson'):np.array([360,  780]),('scorch'):np.array([1625, 737])},
                       'camera-2':{('dodson'):np.array([1500, 565]),('scorch'):np.array([340,  580])}}

boxCorner2_locs_all = {'camera-1':{('dodson'):np.array([360, 320]),('scorch'):np.array([1560, 275])},
                       'camera-2':{('dodson'):np.array([1855,955]),('scorch'):np.array([   9,1010])}}

boxCorner3_locs_all = {'camera-1':{('dodson'):np.array([840,  284]),('scorch'):np.array([1070, 300])},
                       'camera-2':{('dodson'):np.array([1120,1010]),('scorch'):np.array([ 740, 1020])}}

boxCorner4_locs_all = {'camera-1':{('dodson'):np.array([760,    9]),('scorch'):np.array([1140, 3])},
                       'camera-2':{('dodson'):np.array([1155, 310]),('scorch'):np.array([615,320])}}

ndates = np.shape(analyzed_dates)[0]

singlecam_ana_type = "DLC_dlcrnetms5_marmoset_tracking_with_top_cameraMay3shuffle1_150000"
animalnames_videotrack = ['dodson','scorch'] # does not really mean dodson and scorch, instead, indicate animal1 and animal2
bodypartnames_videotrack = ['rightTuft','whiteBlaze','leftTuft','rightEye','leftEye','mouth','lever','tube','boxCorner1','boxCorner2','boxCorner3','boxCorner4']

nanimals_ana = np.shape(animalnames_videotrack)[0]
nbodyparts_ana = np.shape(bodypartnames_videotrack)[0]

do_videodemos = 1

for idate in np.arange(0,ndates,1):
    date_tgt = analyzed_dates[idate]

    twocamera_videos_cam12 = "/gpfs/gibbs/pi/jadi/VideoTracker_SocialInter/test_video_joystick_task_3d/"+date_tgt+"_"+animal1+"_"+animal2+"_camera12/"

    bodyparts_cam1_cam12 = twocamera_videos_cam12+date_tgt+"_"+animal1+"_"+animal2+"_camera-1"+singlecam_ana_type+"_el_filtered.h5"
    bodyparts_cam2_cam12 = twocamera_videos_cam12+date_tgt+"_"+animal1+"_"+animal2+"_camera-2"+singlecam_ana_type+"_el_filtered.h5"

    #bodyparts_3d_cam12_DLC = twocamera_videos_cam12+date_tgt+"_"+animal1+"_"+animal2+"_weikang.h5"

    #
    add_date_dir = twocamera_videos_cam12 +'/anipose_cam123_3d_h5_files/'+date_tgt+'_'+animal1+'_'+animal2
    bodyparts_3d_anipose_file = add_date_dir+'/'+date_tgt+'_'+animal1+'_'+animal2+'_anipose.h5'
    if os.path.exists(bodyparts_3d_anipose_file):
        do_3dconstruct = 0
    else:
        try:
            os.makedirs(add_date_dir)
            do_3dconstruct = 1
        except:
            do_3dconstruct = 1
    do_3dconstruct = 1


    if do_3dconstruct:   
      
        bodyparts_3d_singleAni_anipose_merge = {}
        for ianimal_ana in np.arange(0,nanimals_ana,1):
            
            animalname_ana = animalnames_videotrack[ianimal_ana]
            print('Anipose 3d triangulate witth camera 1, 2 for '+animalname_ana)

            ## save the the h5 file separately for each animals and save them in the same folder for future purpose

            # animal 1 - "dodson" & animal 2 - "scorch"

            # single animal h5 files
            bodyparts_cam1_cam12_singleAni = add_date_dir+'/'+date_tgt+"_"+animal1+"_"+animal2+"_camera-1"+singlecam_ana_type+"_el_filtered_"+animalname_ana+".h5"
            bodyparts_cam2_cam12_singleAni = add_date_dir+'/'+date_tgt+"_"+animal1+"_"+animal2+"_camera-2"+singlecam_ana_type+"_el_filtered_"+animalname_ana+".h5"
            bodyparts_cam3_cam23_singleAni = add_date_dir+'/'+date_tgt+"_"+animal1+"_"+animal2+"_camera-3"+singlecam_ana_type+"_el_filtered_"+animalname_ana+".h5"
  
            # cam1 
            bodyparts_cam1_cam12_data = pd.read_hdf(bodyparts_cam1_cam12)
            bodyparts_cam1_cam12_singleAni_data = {}
            bodyparts_cam1_cam12_singleAni_data[singlecam_ana_type]=bodyparts_cam1_cam12_data.loc[:,(singlecam_ana_type,animalname_ana)]
            bodyparts_cam1_cam12_singleAni_data=pd.concat(bodyparts_cam1_cam12_singleAni_data, axis=1)
            # add lever
            lever_x = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*lever_locs_all['camera-1'][animalname_ana][0]
            lever_y = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*lever_locs_all['camera-1'][animalname_ana][1]
            lever_likelihood = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'lever','x')]=lever_x[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'lever','y')]=lever_y[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'lever','likelihood')]=lever_likelihood[0]
            # add tube
            tube_x = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*tube_locs_all['camera-1'][animalname_ana][0]
            tube_y = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*tube_locs_all['camera-1'][animalname_ana][1]
            tube_likelihood = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'tube','x')]=tube_x[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'tube','y')]=tube_y[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'tube','likelihood')]=tube_likelihood[0]
            # add boxCorner1
            boxCorner1_x = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*boxCorner1_locs_all['camera-1'][animalname_ana][0]
            boxCorner1_y = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*boxCorner1_locs_all['camera-1'][animalname_ana][1]
            boxCorner1_likelihood = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner1','x')]=boxCorner1_x[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner1','y')]=boxCorner1_y[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner1','likelihood')]=boxCorner1_likelihood[0]
            # add boxCorner2
            boxCorner2_x = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*boxCorner2_locs_all['camera-1'][animalname_ana][0]
            boxCorner2_y = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*boxCorner2_locs_all['camera-1'][animalname_ana][1]
            boxCorner2_likelihood = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner2','x')]=boxCorner2_x[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner2','y')]=boxCorner2_y[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner2','likelihood')]=boxCorner2_likelihood[0]
            # add boxCorner3
            boxCorner3_x = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*boxCorner3_locs_all['camera-1'][animalname_ana][0]
            boxCorner3_y = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*boxCorner3_locs_all['camera-1'][animalname_ana][1]
            boxCorner3_likelihood = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner3','x')]=boxCorner3_x[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner3','y')]=boxCorner3_y[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner3','likelihood')]=boxCorner3_likelihood[0]
            # add boxCorner4
            boxCorner4_x = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*boxCorner4_locs_all['camera-1'][animalname_ana][0]
            boxCorner4_y = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))*boxCorner4_locs_all['camera-1'][animalname_ana][1]
            boxCorner4_likelihood = np.ones((1,np.shape(bodyparts_cam1_cam12_singleAni_data)[0]))
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner4','x')]=boxCorner4_x[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner4','y')]=boxCorner4_y[0]
            bodyparts_cam1_cam12_singleAni_data[(singlecam_ana_type,'boxCorner4','likelihood')]=boxCorner4_likelihood[0]
            #      
            bodyparts_cam1_cam12_singleAni_data.to_hdf(bodyparts_cam1_cam12_singleAni,key='tracks')

            # cam2 
            bodyparts_cam2_cam12_data = pd.read_hdf(bodyparts_cam2_cam12)
            bodyparts_cam2_cam12_singleAni_data = {}
            bodyparts_cam2_cam12_singleAni_data[singlecam_ana_type]=bodyparts_cam2_cam12_data.loc[:,(singlecam_ana_type,animalname_ana)]
            bodyparts_cam2_cam12_singleAni_data=pd.concat(bodyparts_cam2_cam12_singleAni_data, axis=1)
            # add lever
            lever_x = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*lever_locs_all['camera-2'][animalname_ana][0]
            lever_y = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*lever_locs_all['camera-2'][animalname_ana][1]
            lever_likelihood = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'lever','x')]=lever_x[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'lever','y')]=lever_y[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'lever','likelihood')]=lever_likelihood[0]
            # add tube
            tube_x = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*tube_locs_all['camera-2'][animalname_ana][0]
            tube_y = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*tube_locs_all['camera-2'][animalname_ana][1]
            tube_likelihood = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'tube','x')]=tube_x[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'tube','y')]=tube_y[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'tube','likelihood')]=tube_likelihood[0]
            # add boxCorner1
            boxCorner1_x = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*boxCorner1_locs_all['camera-2'][animalname_ana][0]
            boxCorner1_y = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*boxCorner1_locs_all['camera-2'][animalname_ana][1]
            boxCorner1_likelihood = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner1','x')]=boxCorner1_x[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner1','y')]=boxCorner1_y[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner1','likelihood')]=boxCorner1_likelihood[0]
            # add boxCorner2
            boxCorner2_x = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*boxCorner2_locs_all['camera-2'][animalname_ana][0]
            boxCorner2_y = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*boxCorner2_locs_all['camera-2'][animalname_ana][1]
            boxCorner2_likelihood = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner2','x')]=boxCorner2_x[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner2','y')]=boxCorner2_y[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner2','likelihood')]=boxCorner2_likelihood[0]
            # add boxCorner3
            boxCorner3_x = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*boxCorner3_locs_all['camera-2'][animalname_ana][0]
            boxCorner3_y = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*boxCorner3_locs_all['camera-2'][animalname_ana][1]
            boxCorner3_likelihood = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner3','x')]=boxCorner3_x[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner3','y')]=boxCorner3_y[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner3','likelihood')]=boxCorner3_likelihood[0]
            # add boxCorner4
            boxCorner4_x = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*boxCorner4_locs_all['camera-2'][animalname_ana][0]
            boxCorner4_y = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))*boxCorner4_locs_all['camera-2'][animalname_ana][1]
            boxCorner4_likelihood = np.ones((1,np.shape(bodyparts_cam2_cam12_singleAni_data)[0]))
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner4','x')]=boxCorner4_x[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner4','y')]=boxCorner4_y[0]
            bodyparts_cam2_cam12_singleAni_data[(singlecam_ana_type,'boxCorner4','likelihood')]=boxCorner4_likelihood[0]
            #      
            bodyparts_cam2_cam12_singleAni_data.to_hdf(bodyparts_cam2_cam12_singleAni,key='tracks')



            ## triangulation without filtering, should take < 15 seconds

            # for single animal
            fname_dict = {
                '1': bodyparts_cam1_cam12_singleAni,
                '2': bodyparts_cam2_cam12_singleAni,
            }
           
            d = load_pose2d_fnames(fname_dict, cam_names=cgroup.get_names())

            score_threshold = 0.1

            n_cams, n_points, n_joints, _ = d['points'].shape
            points = d['points']
            scores = d['scores']

            bodyparts = d['bodyparts']

            # remove points that are below threshold
            points[scores < score_threshold] = np.nan
  
            points_flat = points.reshape(n_cams, -1, 2)
            scores_flat = scores.reshape(n_cams, -1)

            p3ds_flat = cgroup.triangulate(points_flat, progress=True)
            reprojerr_flat = cgroup.reprojection_error(p3ds_flat, points_flat, mean=True)

            p3ds = p3ds_flat.reshape(n_points, n_joints, 3)
            reprojerr = reprojerr_flat.reshape(n_points, n_joints)


            ## save the new h5 files after 3d triangulation
            bodyparts_3d_singleAni_anipose = {}
            
            nbodyparts = np.shape(bodyparts)[0]
            for ibodypart in np.arange(0,nbodyparts,1):
                # remove outlier
                ind_outlier_x = (p3ds[:,ibodypart,0]<(np.nanmean(p3ds[:,ibodypart,0])-2*np.nanstd(p3ds[:,ibodypart,0])))|(p3ds[:,ibodypart,0]>(np.nanmean(p3ds[:,ibodypart,0])+2*np.nanstd(p3ds[:,ibodypart,0])))
                p3ds[ind_outlier_x,ibodypart,0]=np.nan
                ind_outlier_y = (p3ds[:,ibodypart,1]<(np.nanmean(p3ds[:,ibodypart,1])-2*np.nanstd(p3ds[:,ibodypart,1])))|(p3ds[:,ibodypart,1]>(np.nanmean(p3ds[:,ibodypart,1])+2*np.nanstd(p3ds[:,ibodypart,1])))
                p3ds[ind_outlier_y,ibodypart,1]=np.nan
                ind_outlier_z = (p3ds[:,ibodypart,2]<(np.nanmean(p3ds[:,ibodypart,2])-2*np.nanstd(p3ds[:,ibodypart,2])))|(p3ds[:,ibodypart,2]>(np.nanmean(p3ds[:,ibodypart,2])+2*np.nanstd(p3ds[:,ibodypart,2])))
                p3ds[ind_outlier_z,ibodypart,2]=np.nan

                bodyparts_3d_singleAni_anipose[('weikang',animalname_ana,bodyparts[ibodypart],'x')] = p3ds[:,ibodypart, 0]
                bodyparts_3d_singleAni_anipose[('weikang',animalname_ana,bodyparts[ibodypart],'y')] = p3ds[:,ibodypart, 1]
                bodyparts_3d_singleAni_anipose[('weikang',animalname_ana,bodyparts[ibodypart],'z')] = p3ds[:,ibodypart, 2]
            
            bodyparts_3d_singleAni_anipose_merge[ianimal_ana] = pd.DataFrame(bodyparts_3d_singleAni_anipose)


        ## combine the two animals 
 
        bodyparts_3d_anipose = pd.concat([bodyparts_3d_singleAni_anipose_merge[0],bodyparts_3d_singleAni_anipose_merge[1]],axis=1)

        # save the combine the two animal 3d file
        bodyparts_3d_anipose.to_hdf(bodyparts_3d_anipose_file,key='tracks')  

    else:
        ## load saved 3d_anipose h5 file
        bodyparts_3d_anipose = pd.read_hdf(bodyparts_3d_anipose_file)




    ## Section 3 - plot the demo videos
    if do_videodemos:

        print('make the demo Anipose video for '+animal1+' '+animal2)

        import sys
        sys.path.append('/home/ws523/marmoset_tracking_DLCv2/following_up_analysis/3d_recontruction_analysis_joystick_task/ana_functions')
        from tracking_video_3d_demo import tracking_video_3d_demo

        session_start_time = session_start_times[idate]
        fps = 30 
        nframes = 2*fps

        animal1_filename = animal1
        animal2_filename = animal2

        withboxCorner = 1
         
        add_date_dir = twocamera_videos_cam12+'/anipose_cam12_3d_demo_videos/'+date_tgt+'_'+animal1_filename+'_'+animal2_filename
        if not os.path.exists(add_date_dir):
            os.makedirs(add_date_dir)
        video_file = add_date_dir+'/'+date_tgt+'_'+animal1_filename+'_'+animal2_filename+'_anipose_3d_tracking_demo.mp4'
    
        tracking_video_3d_demo(bodyparts_3d_anipose['weikang'],animalnames_videotrack,bodypartnames_videotrack,date_tgt,animal1_filename, animal2_filename,session_start_time,fps,nframes,video_file,withboxCorner)

      
