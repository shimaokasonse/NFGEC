#!/bin/sh

set -o errexit
set -o nounset

echo "Downloading corpus"
wget http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip
unzip corpus.zip
rm corpus.zip

echo "Downloading word embeddings..."
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
mv glove.840B.300d.txt resource/

echo "Preprocessing (creating ids for words, features, and labels)"

echo "OntoNotes"
mkdir ./resource/OntoNotes
python ./resource/create_X2id.py corpus/OntoNotes/all.txt resource/OntoNotes/word2id_gillick.txt resource/OntoNotes/feature2id_gillick.txt resource/OntoNotes/label2id_gillick.txt

echo "Wiki"
mkdir ./resource/Wiki/
python ./resource/create_X2id.py corpus/Wiki/all.txt resource/Wiki/word2id_figer.txt resource/Wiki/feature2id_figer.txt resource/Wiki/label2id_figer.txt

echo "Preprocessing (creating dictionaries)"
mkdir ./data

echo "OntoNotes"
mkdir ./data/OntoNotes
python create_dicts.py resource/OntoNotes/word2id_gillick.txt resource/OntoNotes/feature2id_gillick.txt  resource/OntoNotes/label2id_gillick.txt  resource/glove.840B.300d.txt data/OntoNotes/dicts_gillick.pkl

echo "Wiki"
mkdir ./data/Wiki
python create_dicts.py resource/Wiki/word2id_figer.txt resource/Wiki/feature2id_figer.txt resource/Wiki/label2id_figer.txt  resource/glove.840B.300d.txt data/Wiki/dicts_figer.pkl

echo "Preprocessing (creating datasets)"

echo "OntoNotes"
python create_dataset.py data/OntoNotes/dicts_gillick.pkl corpus/OntoNotes/train.txt data/OntoNotes/train_gillick.pkl
python create_dataset.py data/OntoNotes/dicts_gillick.pkl corpus/OntoNotes/dev.txt data/OntoNotes/dev_gillick.pkl
python create_dataset.py data/OntoNotes/dicts_gillick.pkl corpus/OntoNotes/test.txt data/OntoNotes/test_gillick.pkl

echo "Wiki"
python create_dataset.py data/Wiki/dicts_figer.pkl corpus/Wiki/train.txt data/Wiki/train_figer.pkl
python create_dataset.py data/Wiki/dicts_figer.pkl corpus/Wiki/dev.txt data/Wiki/dev_figer.pkl
python create_dataset.py data/Wiki/dicts_figer.pkl corpus/Wiki/test.txt data/Wiki/test_figer.pkl
