
setlocal
set PYTHONPATH=../..

python ../../main/utils/validation_set_to_prediction.py

call "run_predict"

endlocal




