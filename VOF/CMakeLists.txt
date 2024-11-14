
project(VOF LANGUAGES)

cmake_minimum_required(VERSION 3.20)

find_program(ffmpeg ffmpeg REQUIRED)
find_program(avidemux avidemux3_cli REQUIRED)
find_program(python python REQUIRED)


set(testVideos)
set(horizontalVideos
    "Atoms - 8579.mp4"
    "Frog - 113403.mp4"
    "Grass - 66810.mp4"
    "sea-waves-crashing-on-beach-shore-4793288.mp4"
    "Woman - 58142.mp4"
    "Blue_Sky_and_Clouds_Timelapse_0892__Videvo.mov"
)

set(resolutions 2160;1080;720;360)

# convert all horizontal movies into 4 different resolutions
foreach(resolution ${resolutions})

    set(segments)
    foreach(videoFile ${horizontalVideos})
        get_filename_component(videoName ${videoFile} NAME_WE)
        set(videoFpsFileName "${videoName}-${resolution}p.mp4")
        add_custom_command(
            OUTPUT test-videos/${videoFpsFileName}
            COMMAND ${ffmpeg} -y -i ${CMAKE_CURRENT_SOURCE_DIR}/VOF-media/${videoFile} -filter:v scale=-1:${resolution} test-videos/${videoFpsFileName}
        )
        list(APPEND segments test-videos/${videoFpsFileName})
        set_source_files_properties(test-videos/${videoFpsFileName} PROPERTIES GENERATED TRUE)
    endforeach()

    # combine horizontal files (but Blue Sky) into movies combining all of them (per resolution)
    set(inputFiles)
    set(first TRUE)
    foreach(videoFile ${segments})
        # do not include Blue Sky video in movie, it will be used in movie with intro file
        if(NOT videoFile MATCHES "^test-videos/Blue_Sky_and_Clouds_Timelapse.*")
            if(first)
                set(inputFiles ${inputFiles} --load ${videoFile})
            else()
                set(inputFiles ${inputFiles} --append ${videoFile})
            endif()
            set(first FALSE)
        endif()
    endforeach()

    set(movieFileName movie-${resolution}p.mp4)
    add_custom_command(
        OUTPUT test-videos/${movieFileName}
        COMMAND ${avidemux} ${inputFiles} --video-codec Copy --save test-videos/${movieFileName}
        DEPENDS ${segments}
    )
    list(APPEND testVideos test-videos/${movieFileName})

    # for 720p generate movies with various fps
    if(resolution EQUAL 720)
        set(fpsSet 120;60;30;ntsc;ntsc_film;pal;film;20)
        foreach(fps ${fpsSet})
            set(movieFpsFileName movie-${resolution}p-${fps}_fps.mp4)
            add_custom_command(
                OUTPUT test-videos/${movieFpsFileName}
                COMMAND ${ffmpeg} -y -i test-videos/movie-${resolution}p.mp4 -filter:v fps=fps=${fps} test-videos/${movieFpsFileName}
                DEPENDS test-videos/movie-${resolution}p.mp4
            )
            list(APPEND testVideos test-videos/${movieFpsFileName})
            set_source_files_properties(test-videos/${movieFpsFileName} PROPERTIES GENERATED TRUE)
        endforeach()
    endif()

    # movies with intro
    set(movieWithIntroFileName movie-${resolution}p-w-intro.mp4)
    add_custom_command(
        OUTPUT test-videos/${movieWithIntroFileName}
        COMMAND ${avidemux}
            --load test-videos/Blue_Sky_and_Clouds_Timelapse_0892__Videvo-${resolution}p.mp4
            --append test-videos/movie-${resolution}p.mp4
            --video-codec Copy
            --save test-videos/${movieWithIntroFileName}
        DEPENDS
            test-videos/Blue_Sky_and_Clouds_Timelapse_0892__Videvo-${resolution}p.mp4
            test-videos/movie-${resolution}p.mp4
    )
    list(APPEND testVideos test-videos/${movieWithIntroFileName})

endforeach()


add_custom_target(makeVideos ALL
    DEPENDS ${testVideos}
)


enable_testing()

add_test(
    NAME vof_algo_match_frames
    COMMAND ${python}
            ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/vof_algo_match_frames_tests.py
)

add_test(
    NAME vof_algo_adjust_videos
    COMMAND ${python}
            ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/vof_algo_adjust_videos_tests.py
)

add_test(
    NAME vof_ut_utils
    COMMAND ${python}
            ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/utils_tests.py
)


add_test(
    NAME same_videos_diff_fps
    COMMAND ${python}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/same_videos_different_fps.py
            ${CMAKE_CURRENT_SOURCE_DIR}/vof.py
            ${CMAKE_CURRENT_BINARY_DIR}/test-videos
)

add_test(
    NAME videos_w_and_wo_intro
    COMMAND ${python}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/videos_w_and_wo_intro.py
            ${CMAKE_CURRENT_SOURCE_DIR}/vof.py
            ${CMAKE_CURRENT_BINARY_DIR}/test-videos
)

set_tests_properties(same_videos_diff_fps  PROPERTIES LABELS "IntegrationTest")
set_tests_properties(videos_w_and_wo_intro PROPERTIES LABELS "IntegrationTest")
