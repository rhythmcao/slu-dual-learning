conda create -n slu python=3.6
source activate slu
pip3 install -r requirements.txt

echo "Download pretrained Glove42B 300d word vectors and Kazuma 100d char vectors ..."
mkdir -p ~/.embeddings/glove/
wget -c http://nlp.stanford.edu/data/glove.840B.300d.zip -O ~/.embeddings/glove/common_crawl_840.zip
wget -c https://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz -O ~/.embeddings/kazuma.tar.gz

echo "Download pretrained language model bert-base-uncased into data/.cache/ directory ..."
mkdir -p data/.cache
git lfs install
git clone https://huggingface.co/bert-base-uncased data/.cache/bert-base-uncased