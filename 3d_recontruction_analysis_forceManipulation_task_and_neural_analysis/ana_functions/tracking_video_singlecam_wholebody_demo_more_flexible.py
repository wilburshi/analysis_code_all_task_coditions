#  function - make demo videos for the body part tracking based on single camera, also show the important axes

def tracking_video_singlecam_wholebody_demo_more_flexible(bodyparts_locs_camN, output_look_ornot, output_allvectors, output_allangles, lever_loc_both, tube_loc_both, time_point_pull1, time_point_pull2, animalnames_videotrack, bodypartnames_videotrack, date_tgt, animal1_filename, animal2_filename, session_start_time, fps, nframes, cameraID, video_file_original, sqr_thres_tubelever, sqr_thres_face, sqr_thres_body, start_time=None, end_time=None, speed=1):
    
    # start_time, end_time: session-relative seconds to plot
    # if None, defaults to 0 and nframes/fps



    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import scipy
    import warnings
    import pickle
    import cv2
    import matplotlib.animation as animation
    import os

    skeletons = [
        ['rightTuft','rightEye'],
        ['rightTuft','whiteBlaze'],
        ['leftTuft','leftEye'],
        ['leftTuft','whiteBlaze'],
        ['rightEye','whiteBlaze'],
        ['leftEye','whiteBlaze'],
        ['rightEye','mouth'],
        ['leftEye','mouth'],
        ['leftEye','rightEye']
    ]
    nskeletons = np.shape(skeletons)[0]
    colors     = ['b', 'r', 'k']
    fps        = 30

    # ── time window ──────────────────────────────────────────────────────────
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = nframes / fps

    # absolute frame indices (relative to start of video recording)
    iframe_min = int(np.round((session_start_time + start_time) * fps))
    iframe_max = int(np.round((session_start_time + end_time)   * fps))
    # iframe_max = min(iframe_max, int(np.round(session_start_time * fps)) + nframes)

    # ── video output path ────────────────────────────────────────────────────
    if 0:
        video_file = ("/gpfs/radev/pi/nandy/jadi_gibbs_data/VideoTracker_SocialInter/"
                      "3d_recontruction_analysis_forceManipulation_task_data_saved/"
                      "example_videos_singlecam_wholebody_demo/" + cameraID + "/"
                      + animal1_filename + "_" + animal2_filename + "/"
                      + date_tgt + "_" + animal1_filename + animal2_filename
                      + f"_singlecam_wholebody_tracking_demo_t{int(start_time)}to{int(end_time)}s.mp4")
    if 1:
        video_file = (animal1_filename + "_" + animal2_filename + "/"
                      + date_tgt + "_" + animal1_filename + animal2_filename
                      + f"_singlecam_wholebody_tracking_demo_t{int(start_time)}to{int(end_time)}s.mp4")

    os.makedirs(os.path.dirname(video_file), exist_ok=True)

    # ── try opening video ─────────────────────────────────────────────────────
    try:
        vidcap    = cv2.VideoCapture(video_file_original)
        has_video = vidcap.isOpened()
    except:
        has_video = False
    if not has_video:
        print("No video file found — generating figure without video overlay.")

    # ── writer ────────────────────────────────────────────────────────────────
    FFMpegWriter = animation.writers['ffmpeg']
    metadata     = dict(title='Animal tracking demo', artist='Matplotlib', comment='')
    writer       = FFMpegWriter(fps=fps, metadata=metadata,
                                extra_args=['-vcodec', 'libx264',
                                            '-pix_fmt', 'yuv420p',
                                            '-crf', '23',
                                            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])

    animal_names_unique = animalnames_videotrack
    body_parts_unique   = bodypartnames_videotrack
    nanimals            = np.shape(animal_names_unique)[0]
    nbodyparts          = np.shape(body_parts_unique)[0]

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 16))
    fig.patch.set_facecolor('white')

    # ── helper: make tracking axis ────────────────────────────────────────────
    def make_ax1(gs):
        ax = fig.add_subplot(gs[0:5, :])
        ax.set_xlim([0, 1920])
        ax.set_ylim([0, 1080])
        ax.invert_yaxis()
        ax.axis('off')    # ADD THIS — removes box, ticks, and labels
        return ax

    # ── helper: make event axis ───────────────────────────────────────────────
    def make_event_ax(gs, gs_row, ylabel, show_xticks=False):
        ax = fig.add_subplot(gs[gs_row, :])
        ax.set_xlim([iframe_min, iframe_max])
        ax.set_ylim([0, 1])
        ax.set_ylabel(ylabel, fontsize=14, rotation=0,
                      ha='right', va='center', labelpad=5)
        ax.set_yticklabels('')
        ax.get_yaxis().set_ticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if show_xticks:
            ax.set_xticks(np.arange(iframe_min, iframe_max, 300))
            ax.set_xticklabels(
                list(map(str, np.round(
                    np.arange(start_time, end_time, 300/fps), 1))),
                fontsize=14)
            ax.set_xlabel('time (s)', fontsize=20)
            ax.spines['bottom'].set_visible(True)
        else:
            ax.set_xticklabels('')
            ax.get_xaxis().set_ticks([])
            ax.spines['bottom'].set_visible(False)
        return ax

    # ── render ────────────────────────────────────────────────────────────────
    ax1 = None
    ax2 = None
    ax3 = None
    ax4 = None
    ax5 = None

    iframe_range = np.arange(iframe_min, iframe_max, speed)
    print(f"Total frames to render: {len(iframe_range)}  "
          f"({iframe_min} to {iframe_max})")

    with writer.saving(fig, video_file, 150):
        for iframe in iframe_range:

            print(f"Printing frame {int(iframe)+1} / {iframe_max}", end='\r')

            # ── clear and rebuild axes each frame ─────────────────────────
            fig.clear()
            gs = GridSpec(9, 4, figure=fig, hspace=0.08)

            ax1 = make_ax1(gs)
            ax2 = make_event_ax(gs, 5, 'animal1\ngaze')
            ax3 = make_event_ax(gs, 6, 'animal1\npull')
            ax4 = make_event_ax(gs, 7, 'animal2\ngaze')
            ax5 = make_event_ax(gs, 8, 'animal2\npull',
                                show_xticks=True)

            # start time marker on all event axes
            if 0:
                start_frame_abs = int(np.round((session_start_time + start_time) * fps))
                for ax_ev in [ax2, ax3, ax4, ax5]:
                    ax_ev.axvline(x=start_frame_abs, color='orange',
                                  linewidth=2, linestyle='--', alpha=0.8)

            # current time cursor on all event axes
            for ax_ev in [ax2, ax3, ax4, ax5]:
                ax_ev.axvline(x=iframe, color='gray',
                              linewidth=1, linestyle='--', alpha=0.5)

            # ── video frame ───────────────────────────────────────────────
            if has_video:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, iframe)
                ret, image_original = vidcap.read()
                if ret:
                    ax1.imshow(cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB))
            else:
                ax1.set_facecolor('black')


            # ADD after the video/blank background block:
            t_now = (iframe / fps) - session_start_time
            ax1.text(30, 50, f't = {t_now:.1f} s',
                     color='white', fontsize=20, fontweight='bold', zorder=5,
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3))

            # ── tracking overlay ──────────────────────────────────────────
            for ianimal in np.arange(0, nanimals, 1):
                ianimal_name = animal_names_unique[ianimal]

                # body parts at current frame
                bodypart_loc_iframe = np.zeros((nbodyparts, 2))
                for ibdpart in np.arange(0, nbodyparts, 1):
                    ibdpart_name = body_parts_unique[ibdpart]
                    bodypart_loc_iframe[ibdpart, :] = np.array(
                        bodyparts_locs_camN[(ianimal_name, ibdpart_name)])[iframe, :]

                label = 'animal1' if ianimal == 0 else 'animal2'
                ax1.plot(bodypart_loc_iframe[:, 0], bodypart_loc_iframe[:, 1],
                         '.', color=colors[ianimal], label=label)

                # skeleton
                for iskel in np.arange(0, nskeletons, 1):
                    try:
                        iskeleton_name   = skeletons[iskel]
                        skelbody12       = np.zeros((2, 2))
                        skelbody12[0, :] = np.array(bodyparts_locs_camN[
                            (ianimal_name, iskeleton_name[0])])[iframe, :]
                        skelbody12[1, :] = np.array(bodyparts_locs_camN[
                            (ianimal_name, iskeleton_name[1])])[iframe, :]
                        ax1.plot(skelbody12[:, 0], skelbody12[:, 1],
                                 '-', color=colors[ianimal])
                    except:
                        continue

                # face bounding box
                face_mass = np.nanmean(np.vstack((
                    np.array(bodyparts_locs_camN[(ianimal_name,'rightTuft')])[iframe,:],
                    np.array(bodyparts_locs_camN[(ianimal_name,'whiteBlaze')])[iframe,:],
                    np.array(bodyparts_locs_camN[(ianimal_name,'leftTuft')])[iframe,:],
                    np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:],
                    np.array(bodyparts_locs_camN[(ianimal_name,'mouth')])[iframe,:],
                    np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:])),
                    axis=0)

                dist7  = np.linalg.norm(
                    np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:] -
                    np.array(bodyparts_locs_camN[(ianimal_name,'rightTuft')])[iframe,:])
                dist8  = np.linalg.norm(
                    np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:] -
                    np.array(bodyparts_locs_camN[(ianimal_name,'leftTuft')])[iframe,:])
                dist9  = np.linalg.norm(
                    np.array(bodyparts_locs_camN[(ianimal_name,'whiteBlaze')])[iframe,:] -
                    np.array(bodyparts_locs_camN[(ianimal_name,'rightTuft')])[iframe,:])
                dist10 = np.linalg.norm(
                    np.array(bodyparts_locs_camN[(ianimal_name,'whiteBlaze')])[iframe,:] -
                    np.array(bodyparts_locs_camN[(ianimal_name,'leftTuft')])[iframe,:])
                face_offset = np.nanmax([dist7, dist8, dist9, dist10]) * sqr_thres_face

                # face box
                for xs, ys in [
                    ([face_mass[0]-face_offset, face_mass[0]+face_offset],
                     [face_mass[1]-face_offset, face_mass[1]-face_offset]),
                    ([face_mass[0]-face_offset, face_mass[0]+face_offset],
                     [face_mass[1]+face_offset, face_mass[1]+face_offset]),
                    ([face_mass[0]-face_offset, face_mass[0]-face_offset],
                     [face_mass[1]-face_offset, face_mass[1]+face_offset]),
                    ([face_mass[0]+face_offset, face_mass[0]+face_offset],
                     [face_mass[1]-face_offset, face_mass[1]+face_offset]),
                    # body box
                    ([face_mass[0]-face_offset, face_mass[0]+face_offset],
                     [face_mass[1]-face_offset, face_mass[1]-face_offset]),
                    ([face_mass[0]-face_offset, face_mass[0]+face_offset],
                     [face_mass[1]+sqr_thres_body*face_offset,
                      face_mass[1]+sqr_thres_body*face_offset]),
                    ([face_mass[0]-face_offset, face_mass[0]-face_offset],
                     [face_mass[1]-face_offset,
                      face_mass[1]+sqr_thres_body*face_offset]),
                    ([face_mass[0]+face_offset, face_mass[0]+face_offset],
                     [face_mass[1]-face_offset,
                      face_mass[1]+sqr_thres_body*face_offset]),
                ]:
                    ax1.plot(xs, ys, '--', color=colors[ianimal])

                # head gaze vector
                re      = np.array(bodyparts_locs_camN[(ianimal_name,'rightEye')])[iframe,:]
                le      = np.array(bodyparts_locs_camN[(ianimal_name,'leftEye')])[iframe,:]
                meaneye = np.nanmean(np.vstack([re, le]), axis=0)
                head_tip = meaneye - 400 * np.array(
                    output_allvectors['head_vect_all_merge'][ianimal_name])[iframe,:]
                lbl = 'head gaze' if ianimal == 1 else ''
                ax1.plot([meaneye[0], head_tip[0]],
                         [meaneye[1], head_tip[1]],
                         '-', color='0.75', label=lbl)

                ax1.legend(loc='upper right', fontsize=15)

                # ── behavioral events ──────────────────────────────────────
                look_other = np.where(
                    np.array(output_look_ornot[
                        "look_at_other_or_not_merge"][ianimal_name]) == 1)[0]
                look_other = look_other[
                    (look_other <= iframe) & (look_other > iframe_min)]

                pull1_fn = (time_point_pull1 + session_start_time) * fps
                pull2_fn = (time_point_pull2 + session_start_time) * fps

                if ianimal == 0:
                    for f in look_other:
                        ax2.plot([f, f], [0, 1], '-',
                                 color=colors[0], linewidth=1.5)
                    pull_plot = pull1_fn[
                        (pull1_fn <= iframe) & (pull1_fn > iframe_min)]
                    for f in pull_plot:
                        ax3.plot([f, f], [0, 1], '-',
                                 color='k', linewidth=1.5)

                elif ianimal == 1:
                    for f in look_other:
                        ax4.plot([f, f], [0, 1], '-',
                                 color=colors[1], linewidth=1.5)
                    pull_plot = pull2_fn[
                        (pull2_fn <= iframe) & (pull2_fn > iframe_min)]
                    for f in pull_plot:
                        ax5.plot([f, f], [0, 1], '-',
                                 color='k', linewidth=1.5)

            # ── capture frame ─────────────────────────────────────────────
            fig.canvas.draw()
            writer.grab_frame()

    if has_video:
        vidcap.release()

    plt.close(fig)
    print(f"\nVideo saved → {video_file}")