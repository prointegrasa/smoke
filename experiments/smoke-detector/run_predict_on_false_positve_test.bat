
setlocal

copy .\false-positive-test\*.* .\images-to-predict
call "run_predict"

endlocal




