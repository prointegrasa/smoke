
setlocal


call "run_train"

del .\log\all-checkpoints\*.* /Q
copy .\log\checkpoints\*.* .\log\all-checkpoints

del .\log\checkpoints\model.00*.* /Q
del .\log\checkpoints\model.01*.* /Q
del .\log\checkpoints\model.02*.* /Q
del .\log\checkpoints\model.03*.* /Q
del .\log\checkpoints\model.04*.* /Q
del .\log\checkpoints\model.05*.* /Q
del .\log\checkpoints\model.06*.* /Q
del .\log\checkpoints\model.07*.* /Q
del .\log\checkpoints\model.08*.* /Q

call "run_eval"

endlocal




