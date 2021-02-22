# !/usr/bin bash
#
DOWNLOAD_PATH=~/data/DORi
mkdir -p $DOWNLOAD_PATH
mkdir $DOWNLOAD_PATH/word_embeddings

# download pre-trained spacy models
python -m spacy download en_core_web_md
wget https://cloudstor.aarnet.edu.au/plus/s/ogLztGw9WK2kAWH/download -O preprocessing.zip
unzip preprocessing.zip

cd $DOWNLOAD_PATH
# download pre-trained GLoVe word embeddings
wget -P $DOWNLOAD_PATH/word_embeddings http://nlp.stanford.edu/data/glove.840B.300d.zip 
unzip $DOWNLOAD_PATH/word_embeddings/glove.840B.300d.zip

# download CHARADES i3d features
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/Ac8TJsKYBSFEwhy/download -O charade_sta_i3d_feat.zip
# download CHARADES object features
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/tQ7WYembO3OtgOX/download -O charade_sta_obj.zip

# download YOUCOOKII i3d features
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/P4jmml24WoVH9ev/download -O youcookii_i3d.zip
# download YOUCOOKII object features
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/oAwKQ5xPsurQ1sr/download -O youcookii_obj.z01
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/dLzikggiXVjkg2e/download -O youcookii_obj.z02
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/XHTQYCbX4IxFvHH/download -O youcookii_obj.zip

# download ANET-CAP i3d features.
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/WF0i5SrFqv2NCR1/download
# download ANET-CAP object features.
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/D1bN0TRDPnSX0DY/download -O anetcap_obj.z01
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/nnNN917fOVAvpOS/download -O anetcap_obj.z02 
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/4XYy4mnMcMmI5Cm/download -O anetcap_obj.z03
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/kemsrgg8jC7VLNW/download -O anetcap_obj.z04
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/IWfcmZQnzai3tkG/download -O anetcap_obj.z05
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/fOUWIAlmijTaTWs/download -O anetcap_obj.z06
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/ucw8fsJNSnAgX3h/download -O anetcap_obj.z07
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/YLJcuJXml9klJ8t/download -O anetcap_obj.z08
wget -P $DOWNLOAD_PATH https://cloudstor.aarnet.edu.au/plus/s/sGb2MxVZ4u1ATCv/download -O anetcap_obj.zip
