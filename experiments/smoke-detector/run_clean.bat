
setlocal

del .\training\images\*.* /Q
del .\training\labels\*.* /Q
del .\*.txt /Q

del .\images-raw\*.* /Q

del .\images-to-predict\*.* /Q
del .\images-to-predict-standardized\*.* /Q
del .\prediction-results\*.* /Q

del .\log\checkpoints\*.* /Q
del .\log\tensorboard\*.* /Q
del .\log\tensorboard_val\*.* /Q
del .\log\metrics.csv /Q

endlocal