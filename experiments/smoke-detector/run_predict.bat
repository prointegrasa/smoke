
setlocal

call setenv.bat

dir .\images-to-predict /b /s > images_to_predict.txt

python ../../main/utils/images_raw_to_standardized_for_prediction.py

dir .\images-to-predict-standardized /b /s > images_to_predict.txt

python ../../scripts/predict.py --gpu ""



endlocal