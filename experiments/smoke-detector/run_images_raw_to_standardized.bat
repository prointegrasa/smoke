
setlocal
set PYTHONPATH=../..

dir .\images-raw /b /s > original_images_raw.txt

python ../../main/utils/images_raw_to_standardized.py

endlocal




