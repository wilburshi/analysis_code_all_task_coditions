#  function - get continuous variables and do the fitting

def get_continuous_bhv_var_for_GLM_fitting(animal1, animal2, animalnames_videotrack, min_length, output_look_ornot, output_allvectors, output_allangles, output_key_locations):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle    

    fps = 30 

    nanimals = np.shape(animalnames_videotrack)[0]

    con_vars_plot = ['gaze_other_angle','gaze_tube_angle','gaze_lever_angle','animal_animal_dist','animal_tube_dist','animal_lever_dist','othergaze_self_angle','mass_move_speed','gaze_angle_speed']
    data_summary_names = con_vars_plot

    data_summary = {}


    for ianimal in np.arange(0,nanimals,1):

        animal_name = animalnames_videotrack[ianimal]
        if ianimal == 0:
            animal_name_other = animalnames_videotrack[1]
        elif ianimal == 1:
            animal_name_other = animalnames_videotrack[0]

        # get the variables
        xxx_time = np.arange(0,min_length,1)/fps

        gaze_other_angle = output_allangles['face_eye_angle_all_Anipose'][animal_name]
        # gaze_other_angle = scipy.ndimage.gaussian_filter1d(gaze_other_angle,30) # smooth the curve, use 30 before, change to 3  
        gaze_tube_angle = output_allangles['selftube_eye_angle_all_Anipose'][animal_name]
        # gaze_tube_angle = scipy.ndimage.gaussian_filter1d(gaze_tube_angle,30)  
        gaze_lever_angle = output_allangles['selflever_eye_angle_all_Anipose'][animal_name]
        # gaze_lever_angle = scipy.ndimage.gaussian_filter1d(gaze_lever_angle,30)  

        othergaze_self_angle = output_allangles['face_eye_angle_all_Anipose'][animal_name_other]
        # othergaze_self_angle = scipy.ndimage.gaussian_filter1d(othergaze_self_angle,30)  


        a = output_key_locations['facemass_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['facemass_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_animal_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # animal_animal_dist = scipy.ndimage.gaussian_filter1d(animal_animal_dist,30)  

        a = output_key_locations['tube_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_tube_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # animal_tube_dist = scipy.ndimage.gaussian_filter1d(animal_tube_dist,30)  

        a = output_key_locations['lever_loc_all_Anipose'][animal_name_other].transpose()
        b = output_key_locations['meaneye_loc_all_Anipose'][animal_name].transpose()
        a_min_b = a - b
        animal_lever_dist = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        # animal_lever_dist = scipy.ndimage.gaussian_filter1d(animal_lever_dist,30)  

        a = output_key_locations['facemass_loc_all_Anipose'][animal_name].transpose()
        a = np.hstack((a,[[np.nan],[np.nan],[np.nan]]))
        at1_min_at0 = (a[:,1:]-a[:,:-1])
        mass_move_speed = np.sqrt(np.einsum('ij,ij->j', at1_min_at0, at1_min_at0))*fps 
        # mass_move_speed = scipy.ndimage.gaussian_filter1d(mass_move_speed,30)  

        a = np.array(output_allvectors['eye_direction_Anipose'][animal_name]).transpose()
        a = np.hstack((a,[[np.nan],[np.nan],[np.nan]]))
        at1 = a[:,1:]
        at0 = a[:,:-1] 
        nframes = np.shape(at1)[1]
        gaze_angle_speed = np.empty((1,nframes,))
        gaze_angle_speed[:] = np.nan
        gaze_angle_speed = gaze_angle_speed[0]
        #
        for iframe in np.arange(0,nframes,1):
            gaze_angle_speed[iframe] = np.arccos(np.clip(np.dot(at1[:,iframe]/np.linalg.norm(at1[:,iframe]), at0[:,iframe]/np.linalg.norm(at0[:,iframe])), -1.0, 1.0))    
        # gaze_angle_speed = scipy.ndimage.gaussian_filter1d(gaze_angle_speed,30)  


        # put all the data together in the same order as the con_vars_plot
        if ianimal == 0:
            data_summary[animal1] = [gaze_other_angle,gaze_tube_angle,gaze_lever_angle,animal_animal_dist,animal_tube_dist,animal_lever_dist,othergaze_self_angle,mass_move_speed,gaze_angle_speed]
        elif ianimal == 1:
            data_summary[animal2] = [gaze_other_angle,gaze_tube_angle,gaze_lever_angle,animal_animal_dist,animal_tube_dist,animal_lever_dist,othergaze_self_angle,mass_move_speed,gaze_angle_speed]

    return data_summary, data_summary_names 







########################


def GLM_fitting(animal1, animal2, data_summary, data_summary_names, bhv_data, history_time, nbootstraps, samplesize):

    # nbootstraps: how many repreat of the GLM fitting
    # samplesize: the size of the sample, same for pull events and no-pull events

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle    

    fps = 30

    time_point_pull1 = bhv_data["time_points"][bhv_data["behavior_events"]==1]
    time_point_pull2 = bhv_data["time_points"][bhv_data["behavior_events"]==2]

    # align the pull time to the same temporal resolution
    time_point_pull1 = np.round(time_point_pull1*fps)/fps
    time_point_pull2 = np.round(time_point_pull2*fps)/fps

    nanimals = 2

    for ianimal in np.arange(0,nanimals,1):

        if ianimal == 0:
            data_summary_iani = data_summary[animal1]
            time_point_pulls = time_point_pull1
        elif ianimal == 1:
            data_summary_iani = data_summary[animal2]
            time_point_pulls = time_point_pull2

        # total time point number based on the 30Hz fps videos    
        total_timepoints = np.shape(data_summary_iani)[1]

        event_pulls = np.zeros((1,total_timepoints))[0]
        ind_pulltimepoint = np.array(np.round(time_point_pulls*fps),dtype=int)
        event_pulls[ind_pulltimepoint[ind_pulltimepoint<=total_timepoints]]=1


        # prepare the GLM input and output data

        input_datas = dict.fromkeys(data_summary_names,[])

        output_datas = np.zeros((1,total_timepoints-history_time*fps))[0]

        n_variables = np.shape(data_summary_names)[0]


        for i_timepoint in np.arange(history_time*fps,total_timepoints,1):

            for i_var in np.arange(0,n_variables,1):

                inputdata_xx = np.array(data_summary_iani)[i_var,np.arange(i_timepoint-history_time*fps,i_timepoint,1)]
                if i_timepoint == history_time*fps:
                    input_datas[data_summary_names[i_var]] = inputdata_xx
                else:
                    input_datas[data_summary_names[i_var]] = np.vstack((input_datas[data_summary_names[i_var]],inputdata_xx))

                output_datas[i_timepoint-history_time*fps] = event_pulls[i_timepoint]



 













