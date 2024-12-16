
## Set of tools for batch video file manipulations

Before using any of given scripts backup your data as **source files are being deleted**.

### Merging video files with subtitles into mkv files 

twotone.py is a python script which searches for movie and subtitle files and merges them into one mkv file.<br/>
By default subtitles are added without any language label but it can be changed with \-\-language option. <br/>
See \-\-help for details.


### Automatic video transcoding

transcode.py takes video directory as an input parameter and transcodes all found videos with x265 codec.<br/>
Script tries to find optimal CRF for each video by comparing original video with transcoded one and measuring quality.
As of now it looks for a CRF value giving SSIM ≈ 0.98 result. 

Currently this is a very simple script. It has no 'dry run' mode nor any other options. Just run it with the input dir.<br/>
Please mind ctrl+c is not supported yet, so be careful when you stop its work.
