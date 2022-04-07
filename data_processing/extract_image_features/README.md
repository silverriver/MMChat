# Fast-RCNN feature extractor

You can use the scripts in this folder to extract features for images using the pre-trained Fast-RCNN feature extractor.

## How you use

Run the scripts following this order: 
1. `extract_features_1.py`: extract features from images.
2. `convert_to_lmdb_2.py`: convert the extracted features to the LMDB format to enable fast indexing when utilizing these features.
The pre-trained Fast-RCNN model can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/13AbkX4JjrHXx-4MNF_4a6w?pwd=vxs5).
Before running the above script, you need to install the `maskrcnn_benchmark` lib with the following steps:
```bash
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
git checkout 4c168a637f45dc69efed384c00a7f916f57b25b8 -b stable
python setup.py develop
```

-----

Or, you can use the script provided by [Facebook ParlAI](https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/image_featurizers.py) to extract the features.
Specifically, if you instantiate an ImageLoader with the “faster_r_cnn_152_32x8d” image mode, you can simply call loader.extract to load the images.
