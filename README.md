# Adaptation of RelayNet

Original authors : Abhijit Guha Roy and Shayan Siddiqui (https://github.com/shayansiddiqui)

Adapted by: Charlene Ong

## Table of Contents
1. [Description](#description)
2. [About the Files](#about-files)
3. [Directory Structure](#directory-structure)
4. [Datasets](#datasets)
5. [How to Run the Codes](#run-codes)
6. [References](#references)
7. [License](#license)

## Description <a name="description"></a>

PyTorch Implementation of ReLayNet. The original implementation came from the original github repo[https://github.com/ai-med/relaynet_pytorch] but some modifications are made by yours truly for the Data Resource for Healthy Controls and Multiple Sclerosis dataset. 

## About the Files <a name="about-files"></a>
1. `networks`: folder containing the model architecture, training, losses 
2. `preprocess`: folder to preprocess datasets.
3. `train.py`: train the model. Configuration file is `train.yaml` and identical jupyter notebook is `train.ipynb`
4. `test.py`: test on test datasets. Configuration file is `test.yaml` and identical jupyter notebook is `test.ipynb`

## Directory Structure <a name="directory-structure"></a>
```bash
├── src
│   ├── __init__.py
│   ├── networks
│   │   ├── net_api
│   │   ├── __init__.py
│   │   ├── data_utils.py
│   │   ├── relay_net.py
│   │   └── solver.py
│   ├── preprocess
│   │   ├── data_prep_utils
│   │   ├── __init__.py
│   │   ├── preprocess_config_JH.yaml
│   │   └── preprocess_datasets.ipynb
│   ├── test.py
│   ├── test.ipynb
│   ├── train.py
│   ├── train.ipynb
│   ├── test.yaml
│   └── train.yaml
├── data
│   ├── labels
│   ├── processed
│   └── raw
├── evaluations
│   ├── JH
│   ├── MIAMI_HC
│   └── ...
├── models
│   ├── Exp01
│   ├── Exp02
│   └── ...
├── predictions
    ├── hc05_spectralis_macula_v1_s1_R
    ├── hc07_spectralis_macula_v1_s1_R
    └── hc09_spectralis_macula_v1_s1_R
    
```

## How to Run the Codes <a name="run-codes"></a>
1. Modify `./preprocess/preprocess_config_JH.yaml` according to your needs. Run `preprocess_dataset.ipynb`.
2. Modify `./train.yaml` according to your needs. Run 
```bash
python train.py
```
3. Modify `./test.yaml` according to your needs. Run
```bash
python test.py
```

## References <a name="references"></a>

````bibtex
A. Guha Roy, S. Conjeti, S.P.K.Karri, D.Sheet, A.Katouzian, C.Wachinger, and N.Navab, "ReLayNet: retinal layer and fluid segmentation of macular optical coherence tomography using fully convolutional networks," Biomed. Opt. Express 8, 3627-3642 (2017) 
Link: https://arxiv.org/abs/1704.02161
````

## License <a name="license"></a>
License by original authors, Abhijit Guha Roy and Shayan Siddiqui
___

Original MIT License by Authors

MIT License

Copyright (c) 2018 Abhijit Guha Roy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


