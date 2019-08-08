install-linux:
	sudo pip install -r requirements.txt
	sudo apt-get install texlive-latex-base
	sudo apt-get install texlive-latex-extra

	sudo apt-get update
	sudo apt-get install ghostscript
	sudo apt-get install libgs-dev

	wget http://www.imagemagick.org/download/ImageMagick.tar.gz
	tar -xvf ImageMagick.tar.gz
	cd ImageMagick-7.*; \
	./configure --with-gslib=yes; \
	make; \
	sudo make install; \
	sudo ldconfig /usr/local/lib
	cd ..
	rm ImageMagick.tar.gz
	rm -r ImageMagick-7.*

install-mac:
	sudo pip install -r requirements.txt
	wget http://www.imagemagick.org/download/ImageMagick.tar.gz
	tar -xvf ImageMagick.tar.gz
	cd ImageMagick-7.*; \
	./configure --with-gslib=yes; \
	make;\
	sudo make install; \
	cd ..
	rm ImageMagick.tar.gz
	rm -r ImageMagick-7.*

build-small:
	python build.py --data=configs/data_small.json --vocab=configs/vocab_small.json

train-small:
	python train.py --data=configs/data_small.json --vocab=configs/vocab_small.json --training=configs/training_small.json --model=configs/model.json --output=results/small/

eval-small:
	python evaluate_txt.py --results=results/small/
	python evaluate_img.py --results=results/small/

small: build-small train-small eval-small

build:
	python build.py --data=configs/data.json --vocab=configs/vocab.json

train:
	python train.py --data=configs/data.json --vocab=configs/vocab.json --training=configs/training.json --model=configs/model.json --output=results/full/

eval:
	python evaluate_txt.py --results=results/full/
	python evaluate_img.py --results=results/full/

full: build train eval
