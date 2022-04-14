# MMChat

This repo contains the code and data for the LREC2022 paper 
**[MMChat: Multi-Modal Chat Dataset on Social Media](https://arxiv.org/abs/2108.07154)**.

## Dataset

MMChat is a large-scale dialogue dataset that contains image-grounded dialogues in Chinese.
Each dialogue in MMChat is associated with one or more images (maximum 9 images per dialogue).
We design various strategies to ensure the quality of the dialogues in MMChat. Please read our paper for more details.
The images in the dataset are hosted on Weibo's static image server. 
You can refer to the scripts provided in `data_processing/weibo_image_crawler` to download these images.

Two sample dialogues form MMChat are given below (translated from Chinese):
![A sample dialogue from MMChat](/bin/sample.jpg)

MMChat is released in different versions:

### Rule Filtered Raw MMChat

This version of MMChat contains raw dialogues filtered by our rules.
The following table shows some basic statistics:

| Item Description                     | Count    |
|--------------------------------------|---------:|
| Sessions                             | 4.257 M  |
| Sessions with more than 4 utterances | 2.304 M  |
| Utterances                           | 18.590 M |
| Images                               | 4.874 M  |
| Avg. utterance per session           | 4.367    |
| Avg. image per session               | 1.670    |
| Avg. character per utterance         | 14.104   |

We devide above dialogues into 9 splits to facilitate the download:

0. Split0 [Google Drive](https://drive.google.com/file/d/1irGoKFDqorFNwZtySrA1-g12dl61pG-7/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1JJ627hzIDG1c4gxbZQcbRg?pwd=mviv)
1. Split1 [Google Drive](https://drive.google.com/file/d/1OkpF7MAtntn2czuZfujSRc_7rALJ6VRJ/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1iupSNrqUd4pQVESOFqNmyw?pwd=ocqr)
2. Split2 [Google Drive](https://drive.google.com/file/d/1pv_NsPNdQrBSve3h9eVRH1MjeBH8w1AF/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1iX10kUf1at1sCUU83b8SmA?pwd=4f88)
3. Split3 [Google Drive](https://drive.google.com/file/d/14OSOAD7gM6nVa1ydwJTSApM2WzGOBcWV/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1cq0O1QITtykhB8L0MUlqtw?pwd=w3v5)
4. Split4 [Google Drive](https://drive.google.com/file/d/14Fz2kof5CBjdgyabxZ8hS6g1hN-9owLx/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1snRfnNN4kbGzfxhbcNFe3g?pwd=xzg9)
5. Split5 [Google Drive](https://drive.google.com/file/d/1xKAzn9oeWewBKHIb3bt14g4gnrO0u2CP/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1APwm7xTE2oID92Xb74q6Zw?pwd=vvsx)
6. Split6 [Google Drive](https://drive.google.com/file/d/1vbf8piV9hSCyo2pvx91W4lynZhKNx2lM/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/10HV3p3wnLhHHFOdbhJJOSg?pwd=5idw)
7. Split7 [Google Drive](https://drive.google.com/file/d/1qfQ3c7SoR44Xd-4HfBb-wh_GOArUhyBz/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1BOSTdHzQizZAMavy1aeajg?pwd=yx6q)
8. Split8 [Google Drive](https://drive.google.com/file/d/1J4LvdVyX83YsMKh04CeIfTF1N13Q3d3N/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/11VQL7rUrJtmp74x97C5L6g?pwd=lu0i)

### LCCC Filtered MMChat

This version of MMChat contains the dialogues that are filtered based on the [LCCC](https://github.com/thu-coai/CDial-GPT) (Large-scale Cleaned Chinese Conversation) dataset.
Specifically, some dialogues in MMChat are also contained in LCCC. 
We regard these dialogues as cleaner dialogues since sophisticated schemes are designed in LCCC to filter out noises.
This version of MMChat is obtained using the script `data_processing/LCCC_filter.py`
The following table shows some basic statistics:

| Item Description                     | Count   |
|--------------------------------------|--------:|
| Sessions                             | 492.6 K |
| Sessions with more than 4 utterances | 208.8 K |
| Utterances                           | 1.986 M |
| Images                               | 1.066 M |
| Avg. utterance per session           | 4.031   |
| Avg. image per session               | 2.514   |
| Avg. character per utterance         | 11.336  |

We devide above dialogues into 9 splits to facilitate the download:

0. Split0 [Google Drive](https://drive.google.com/file/d/1Qd3N00ZpVOGDBqwlHcpj_QgbSIYNnysx/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/17g0UBF8zT3w5hfzvpYerQA?pwd=b2an)
1. Split1 [Google Drive](https://drive.google.com/file/d/1H15T_aSLNaLZdc86WsUU6-c0J37OoZW-/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1xj_RIE60Be-sisdkrWt0fQ?pwd=6z1x)
2. Split2 [Google Drive](https://drive.google.com/file/d/1dCXlyQGwx5tfRFLnsDp0B5LhdHr_Rsbi/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1_0WFHqK1ZY92yC4BEqRSwQ?pwd=35cw)
3. Split3 [Google Drive](https://drive.google.com/file/d/1jzLgo2JW87cjGxEMRtKC8KorTIv-ODJR/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1_pgQRtr7LYnH0aQagRr2Bg?pwd=ouo0)
4. Split4 [Google Drive](https://drive.google.com/file/d/1JiGhdzhzMZhL_dGreZclymhHxE7YuiRy/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/128CzlJpqKxhc4GJeRynX-g?pwd=pnmr)
5. Split5 [Google Drive](https://drive.google.com/file/d/1ZLdsNZyFG-cq9pqHP5KvfL0fPXqmmXxO/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1Y21T3jMPWSiRCATvYNOC4g?pwd=ca3m)
6. Split6 [Google Drive](https://drive.google.com/file/d/1qi99_TFwJanuGgAWDBRgi6hqNUQB9JQd/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1hfBchNqVhOYjFk9fTT_gxA?pwd=dzh3)
7. Split7 [Google Drive](https://drive.google.com/file/d/15QMZhGuW93fzAVRhBKb6ANiZ8BNw5lX9/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1amg65X0ST7gW8c8MCutXWQ?pwd=2t1j)
8. Split8 [Google Drive](https://drive.google.com/file/d/1wRCiJfxNk5n5SYzMBm4HYM1BKyGtuGak/view?usp=sharing), [Baidu Netdisk](https://pan.baidu.com/s/1-KYwR-SOezyn5jFzrA3Fxw?pwd=0pyi)

### MMChat

The MMChat dataset reported in our paper are given here.
The Weibo content corresponding to these dialogues are all "分享图片", (i.e., "Share Images" in English).
The following table shows some basic statistics:

| Item Description                     | Count   |
|--------------------------------------|--------:|
| Sessions                             | 120.84 K |
| Sessions with more than 4 utterances |  17.32 K |
| Utterances                           | 314.13 K |
| Images                               |  198.82 K |
| Avg. utterance per session           |  2.599 |
| Avg. image per session               |  2.791 |
| Avg. character per utterance         |  8.521 |

The above dialogues can be downloaded from either [Google Drive](https://drive.google.com/drive/folders/1sBzuJzOpPEj6-IoXl3drvfqQ8i1_tluX?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1m9nwZejujNUIcVUiIKcxPg?pwd=nrqr).

### MMChat-hf

We perform human annotation on the sampled dialogues to determine whether the given images are related to the corresponding dialogues.
The following table only shows the statistics for dialogues that are annotated as image-related.

| Item Description                     | Count   |
|--------------------------------------|--------:|
| Sessions                             | 19.90 K |
| Sessions with more than 4 utterances | 8.91 K |
| Utterances                           | 81.06 K |
| Images                               | 52.66K |
| Avg. utterance per session           | 4.07 |
| Avg. image per session               | 2.70 |
| Avg. character per utterance         | 11.93 |

We annotated about 100K dialogues.
All the annotated dialogues can be downloaded from either [Google Drive](https://drive.google.com/drive/folders/1dGg4Coc4bwH7tk7SWn0quTwMYxn-kX70?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/11l-bYAKoLkm4k7zDPrfZvg?pwd=zfw2).

## Code 

We are also releasing all the codes used for our experiments.
You can use the script `run_training.sh` in each folder to launch the distributed training.

For models that require image features, you can extract the image features using the scripts in `data_processing/extract_image_features`

The model shown in our paper can be found in `dialog_image`:
![Model](/bin/model.jpg)

## Reference
Please cite our paper if you find our work useful ;)

```bibtex
@inproceedings{zheng2022MMChat,
  author    = {Zheng, Yinhe and Chen, Guanyi and Liu, Xin and Sun, Jian},
  title     = {MMChat: Multi-Modal Chat Dataset on Social Media},
  booktitle = {Proceedings of The 13th Language Resources and Evaluation Conference},
  year      = {2022},
  publisher = {European Language Resources Association},
}
```

```bibtex
@inproceedings{wang2020chinese,
  title     = {A Large-Scale Chinese Short-Text Conversation Dataset},
  author    = {Wang, Yida and Ke, Pei and Zheng, Yinhe and Huang, Kaili and Jiang, Yong and Zhu, Xiaoyan and Huang, Minlie},
  booktitle = {NLPCC},
  year      = {2020},
  url       = {https://arxiv.org/abs/2008.03946}
}
```