

download: # Target to download the LibriTTS dataset
	python src/setup/LibriTTS_download.py

extract: # Target to extract the LibriTTS dataset
	python src/setup/LibriTTS_extract.py

convert: # Target to convert the LibriTTS dataset
	python src/setup/LibriTTS_convert.py

setup_all: # Target to perform all setup steps
	$(MAKE) download
	$(MAKE) extract
	$(MAKE) convert