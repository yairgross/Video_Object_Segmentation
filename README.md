
# Streaming Data Clustering for Object Segmentation
This project focuses on the task of clustering streaming data, specifically applied to video object segmentation.
Our project aims to track objects in video streams by dividing frames into object and background clusters. We built upon ScStream code written as the foundation for our work.

## Program Description
Our project achieves video object segmentation through the following steps:

**Data Masking**: The masking.py script extracts the initial frame from the video and applies a binary mask to create object and background images.

**Data Transformation**: The object_segmentation.py script transforms the images into pixel matrices, with each row representing pixel data, including location and color channels. This transformation enhances data for clustering.

**Clustering**: We use a package of parallel streaming clustering code to create object and background models, updating them for each frame of the video.

**Object Segmentation**: The result is an output video (final_output) that highlights the segmented object throughout the video.

## Results
To demonstrate our project's performance, we provide the following:

**final_output**: The output video highlighting the segmented object.

**comparing_video**: A side-by-side comparison of the input and output videos for visual evaluation.




The following is a demonstration of the output video, alongside the original video:

![output_example](https://github.com/yairgross/Video_Object_Segmentation/blob/main/Project/output_short.gif)https://github.com/yairgross/Video_Object_Segmentation/blob/main/Project/output_short.gif
