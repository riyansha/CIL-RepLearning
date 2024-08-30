# FsACLearning

The official PyTorch implementation of our **Interspeech** 2024 paper:

**Towards Robust Few-shot Class Incremental Learning in Audio Classification using Contrastive Representation** [[paper]](http://arxiv.org/abs/2407.19265)

## Abstract
In machine learning applications, gradual data ingress is common, especially in audio processing where incremental learning is vital for real-time analytics. Few-shot class-incremental learning addresses challenges arising from limited incoming data. Existing methods often integrate additional trainable components or rely on a fixed embedding extractor post-training on base sessions to mitigate concerns related to catastrophic forgetting and the dangers of model overfitting. However, using cross-entropy loss alone during base session training is suboptimal for audio data. To address this, we propose incorporating supervised contrastive learning to refine the representation space, enhancing discriminative power and leading to better generalization since it facilitates seamless integration of incremental classes, upon arrival. Experimental results on NSynth and LibriSpeech datasets with 100 classes, as well as ESC dataset with 50 and 10 classes, demonstrate state-of-the-art performance.

## Pipline
The whole learning pipline of our model:

<img src='imgs/pipeline.png' width='900' height='454'>


## Strong Performance
### Results obtained by various methods on LS-100 Dataset
Our proposed method steadily and dramatically outperforms the state-of-the-art methods.
"Base," "Incr.," and "All" denotes the performance (accuracy) for base, incremental, and combined sessions.

| Methods | Session | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | AA (%) | PD (%) |
|---------|---------|------|------|------|------|------|------|------|------|------|--------|--------|
| FT      | Base    | 92.02| 72.90| 37.03| 28.12| 20.75| 14.45| 5.70 | 3.23 | 0.27 | 30.50  | 91.75  |
|         | Incr.   | -    | 86.60| 31.50| 28.87| 25.45| 24.24| 18.17| 13.46| 11.80| 30.01  | 74.80  |
|         | All     | 92.02| 73.95| 36.24| 28.27| 21.93| 17.33| 9.86 | 7.00 | 4.88 | 32.39  | 87.14  |
| iCaRL   | Base    | 92.02| 80.80| 73.18| 58.45| 26.95| 16.93| 32.58| 29.53| 26.38| 48.54  | 65.64  |
|         | Incr.   | -    | 58.00| 67.10| 57.40| 20.05| 16.48| 30.33| 26.83| 28.95| 38.14  | 29.05  |
|         | All     | 92.02| 79.05| 72.31| 58.24| 25.23| 16.80| 31.83| 28.54| 27.41| 47.94  | 64.61  |
| DFSL    | Base    | 91.93| 91.93| 91.88| 91.85| 91.83| 91.86| 91.85| 91.85| 91.84| 91.87  | 0.09   |
|         | Incr.   | -    | 53.60| 61.90| 50.67| 48.90| 51.56| 47.97| 44.11| 45.38| 50.51  | 8.22   |
|         | All     | 91.93| 88.97| 87.60| 83.61| 81.11| 80.01| 77.22| 74.26| 73.25| 81.99  | 18.68  |
| CEC     | Base    | 91.72| 91.67| 91.25| 91.14| 91.10| 91.07| 90.97| 90.66| 90.72| 91.14  | 1.00   |
|         | Incr.   | -    | 86.30| 82.76| 69.67| 68.25| 67.06| 66.03| 60.35| 60.05| 70.06  | 26.25  |
|         | All     | 91.72| 91.25| 90.04| 86.84| 85.38| 84.01| 82.65| 79.49| 78.45| 85.54  | 13.27  |
| SC      | Base    | 92.73| 92.72| 92.62| 92.48| 92.48| 92.47| 92.34| 90.74| 90.67| 92.14  | 2.06   |
|         | Incr.   | -    | 86.84| 84.26| 77.74| 74.99| 75.79| 74.60| 72.45| 72.64| 77.41  | 14.20  |
|         | All     | 92.73| 92.27| 91.42| 89.53| 88.10| 87.56| 86.43| 84.00| 83.45| 88.39  | 9.28   |
| Ours    | Base    | 92.97| 92.80| 92.37| 91.50| 91.58| 91.90| 91.70| 91.03| 90.88| 91.86  | 2.09   |
|         | Incr.   | -    | 99.60| 97.00| 92.73| 91.05| 89.64| 89.43| 86.14| 85.63| 91.40  | 13.97  |
|         | All     | **92.97**| **93.32**|**93.03**|**91.75**|**91.46**|**91.24**|**90.95**|**89.23**|**88.78**|**91.41**|**4.18**|


## Datasets

To study the Few-shot Class-incremental Audio Classification (FCAC) problem, LS-100 dataset and NSynth-100 dataset are constructed by choosing samples from audio corpora of the [Librispeech](https://www.openslr.org/12/) dataset, the [NSynth](https://magenta.tensorflow.org/datasets/nsynth) dataset respectively. Wei Xie, one of our team members, constructed the NSynth-100 dataset

The detailed information of the LS-100 dataset and NSynth-100 dataset are given below.

### Statistics on the LS-100 dataset

|                                                                 | LS-100                                        | NSynth-100                                    |
| --------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| Type of audio                                                   | Speech                                        | Musical instrumnets                           |
| Num. of classes                                                 | 100 (60 of base classes, 40 of novel classes) | 100 (55 of base classes, 45 of novel classes) |
| Num. of training / validation / testing samples per base class  | 500 / 150 / 100                               | 200 / 100 / 100                               |
| Num. of training / validation / testing samples per novel class | 500 / 150 / 100                               | 100 / none / 100                              |
| Duration of the sample                                          | All in 2 seconds                              | All in 4 seconds                              |

##### Preparation of the NSynth-100 dataset

The NSynth dataset is an audio dataset containing 306,043 musical notes, each with a unique pitch, timbre, and envelope. 
Those musical notes are belonging to 1,006 musical instruments.

Based on the statistical results, we obtain the NSynth-100 dataset by the following steps:

1. Download [Train set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz), [Valid set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz), and [test set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz) of the NSynth dataset to your local machine and unzip them.

2. Download the meta files for FCAC from [here](./data/nsynth) to your local machine and unzip them.

3. You will get s structure of the directory as follows:

   ```
   Your dataset root(Nsynth)
   ├── nsynth-100-fs-meta
   ├── nsynth-200-fs-meta
   ├── nsynth-300-fs-meta
   ├── nsynth-400-fs-meta
   ├── nsynth-test
   │   └── audio
   ├── nsynth-train
   │   └── audio
   └── nsynth-valid
       └── audio
   ```

   

### Preparation of the LS-100 dataset

LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned. We find that the subset `train-clean-100` of Librispeech is enough for our study, so we constructed the LS-100 dataset using partial samples from the Librispeech as the source materials. To be specific, we first concatenate all the speakers' speech clips into a long speech, and then select the 100 speakers with the longest duration to cut their voices into two second speech. You can download the Librispeech from [here](https://www.openslr.org/12/).

1. Download [dataset](https://www.openslr.org/resources/12/train-clean-100.tar.gz) and extract the files.

2. Transfer the format of audio files. Move the script `normalize-resample.sh` to the root dirctory of extracted folder, and run the command `bash normalize-resample.sh`.

3. Construct LS-100 dataset.
   
   ```
   python data/LS100/construct_LS100.py --data_dir DATA_DIR --duration_json data/librispeech/spk_total_duration.json --single_spk_dir SINGLE_SPK_DIR --num_select_spk 100 --spk_segment_dir SPK_SEGMENT_DIR --csv_path CSV_PATH --spk_mapping_path SPK_MAPPING_PATH
   ```

## Training scripts

- Nsynth

  ```
  python3 train.py -project meta_sc -dataroot /data1/nsynth/ -dataset nsynth-100 -lamda_proto 0.6  -num_session 10 -batch_size_base 128 -config ./configs/meta_sc_NS-100_stochastic_classifier.yml -gpu 0 
  ```
  
- LibriSpeech
    ```
    python train.py -project meta_sc -dataroot /data1/LibriSpeech_fscil/ffmpeg-6.1-amd64-static/spk_segments -dataset librispeech -batch_size_base 128 -lamda_proto 0.6 -num_session 9 -config ./configs/meta_sc_LS-100_stochastic_classifier.yml -gpu 0
    ```

- ESC50
    ```
    python3 train.py -project meta_sc -dataroot /data1/ESC-50/archive/audio/audio/ -dataset esc -lamda_proto 0.6 -num_session 10 -way 2 -shot 2 -batch_size_base 128 -config ./configs/meta_sc_ES-100_stochastic_classifier.yml -gpu 0
    ```

- ESC10    
     ```
    python3 train.py -project meta_sc -dataroot /data1/ESC-10/archive/Data/ -dataset pqrs -lamda_proto 0.6 -num_session 3 -batch_size_base 128 -config ./configs/meta_sc_ES-10_stochastic_classifier.yml -gpu 0
    ```

  

## Acknowledgment

Our project references the codes in the following repos.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [SC](https://github.com/vinceasvp/meta-sc)

## Citation

Please consider giving us a citation and a star to our repository if any portion of our paper and code is useful for your research.

```
@misc{singh2024robustfewshotclassincremental,
    title={Towards Robust Few-shot Class Incremental Learning in Audio Classification using Contrastive Representation}, 
    author={Riyansha Singh and Parinita Nema and Vinod K Kurmi},
    year={2024},
    eprint={2407.19265},
    archivePrefix={arXiv},
    primaryClass={cs.SD},
    url={https://arxiv.org/abs/2407.19265}, 
}
```
