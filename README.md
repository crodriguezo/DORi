 

# DORi: Discovering Object Relationships for Moment Localization of a Natural Language Query in a Video Code accompanying the paper 

This repository includes:

* Code for training and testing our model for temporal moment localization [DORi: Discovering Object Relationships for Moment Localization of a Natural](https://openaccess.thecvf.com/content/WACV2021/papers/Rodriguez-Opazo_DORi_Discovering_Object_Relationships_for_Moment_Localization_of_a_Natural_WACV_2021_paper.pdf)
* Links to the I3D and object features we extracted for the YouCookII, Charades-STA and Activity-Net which were used for the experiments in our paper.
* Links to pre-trained models on the YouCookII, Charades-STA and Activity-Net datasets. 

# Installation

1. Clone this repo                                                                                                              
   ```bash
   git clone https://github.com/crodriguezo/DORi.git
   cd DORi
   ```

2. Create a conda environment based on our dependencies and activate it

   ```bash
   conda create -n <name> --file environment.txt
   conda activate <name>
   ```

   Where you can replace `<name>` by whatever you want.

2. Download everything
   ```bash
   sh ./download.sh
   ```
   This script will download the following things in the folder `~/data/DORi`: 
   * The `glove.840B.300d.txt` pre-trained word embeddings.
   * The I3D features for Charades-STA, YouCookII and Activity-Net we extracted and used in our experiments.

   This script will also install the `en_core_web_md` pre-trained spacy model, and download the pre-processed annotations.

   Downloading everything can take a while depending on your internet connection, please be patient. 

## Configuration
 If you have modified the download path from the defaults in the script above please modify the contents of the file `./config/settings.py` accordingly.

# Training

To train our model in the Charades-STA dataset, please run:
```bash
python main.py --config-file=experiments/CharadesSTA/CharadesSTA_train.yaml
```
# Testing

To load our pre-trained model and test it, first make sure the weights have been downloaded and are in the `./checkpoints/charades_sta` folder. Then simply run:

```bash
python main.py --config-file=experiments/CharadesSTA/CharadesSTA_test.yaml
```

# Download Links

If you are interested in downloading some specific resource only, we provide the links below.

**I3D Features**

* [Charades-STA](https://cloudstor.aarnet.edu.au/plus/s/Ac8TJsKYBSFEwhy/download)
* [YouCookII](https://cloudstor.aarnet.edu.au/plus/s/P4jmml24WoVH9ev/download)
* [Activity-Net](https://cloudstor.aarnet.edu.au/plus/s/WF0i5SrFqv2NCR1/download)

**Object Features**

* [Charades-STA](https://cloudstor.aarnet.edu.au/plus/s/tQ7WYembO3OtgOX/download)
* YouCookII [1](https://cloudstor.aarnet.edu.au/plus/s/oAwKQ5xPsurQ1sr/download), [2](https://cloudstor.aarnet.edu.au/plus/s/dLzikggiXVjkg2e/download), [3](https://cloudstor.aarnet.edu.au/plus/s/XHTQYCbX4IxFvHH/download)
* Activity-Net [1](https://cloudstor.aarnet.edu.au/plus/s/D1bN0TRDPnSX0DY/download), [2](https://cloudstor.aarnet.edu.au/plus/s/nnNN917fOVAvpOS/download), [3](https://cloudstor.aarnet.edu.au/plus/s/4XYy4mnMcMmI5Cm/download), [4](https://cloudstor.aarnet.edu.au/plus/s/kemsrgg8jC7VLNW/download), [5](https://cloudstor.aarnet.edu.au/plus/s/IWfcmZQnzai3tkG/download), [6](https://cloudstor.aarnet.edu.au/plus/s/fOUWIAlmijTaTWs/download), [7](https://cloudstor.aarnet.edu.au/plus/s/ucw8fsJNSnAgX3h/download), [8](https://cloudstor.aarnet.edu.au/plus/s/YLJcuJXml9klJ8t/download), [9](https://cloudstor.aarnet.edu.au/plus/s/sGb2MxVZ4u1ATCv/download)

**GLoVe**

* [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)

**Pretrained weights**

* [Charades-STA](https://cloudstor.aarnet.edu.au/plus/s/eSa3t3sWMV0DbWT/download)
* [YouCookII](https://cloudstor.aarnet.edu.au/plus/s/gigu0M5Jt4cc7FR/download)
* [Activity-Net](https://cloudstor.aarnet.edu.au/plus/s/du2PQvf9G8FKZkm/download)


# Citation

If you use our code or features please consider citing our works.

```bibtex
@InProceedings{rodriguez2020proposal,
    title={Proposal-free Temporal Moment Localization of a Natural-Language Query in Video using Guided Attention},
    author={Rodriguez-Opazo, Cristian and Marrese-Taylor, Edison and Saleh, Fatemeh Sadat and Li, Hongdong and Gould, Stephen},
    booktitle={2020 IEEE Winter Conference on Applications of Computer Vision (WACV)},
    pages={2453--2462},
    year={2020},
    organization={IEEE}
}

@InProceedings{Rodriguez-Opazo_2021_WACV,
    title     = {DORi: Discovering Object Relationships for Moment Localization of a Natural Language Query in a Video},
    author    = {Rodriguez-Opazo, Cristian and Marrese-Taylor, Edison and Fernando, Basura and Li, Hongdong and Gould, Stephen},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {1079-1088}
}
```


​    
