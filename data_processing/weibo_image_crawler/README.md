# Scripts used to download image

The MMChat dataset contains dialogues and their corresponding images.
However, we do not host these images due to the pravicy issue.
We only provide the urls to each image so that users of MMChat can download these images by themselves.
You can use the scripts in this folder to download images.

## How to use

Run the scripts following this order: 
1. `remove_dup_weiboid_1.py`: skip weibo id that has been handled.
2. `download_weibo_img_2.py`: download images based on the given url.
3. `filter_broken_url_3.py`: postprocess the image index file and remove the images that are not downloaded.

Please read each script to find out the necessary augments for each script.
