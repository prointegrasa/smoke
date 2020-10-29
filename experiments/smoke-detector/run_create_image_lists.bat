
setlocal
set PYTHONPATH=../..

dir .\training\images /b /s > images.txt
dir .\training\labels /b /s > labels.txt

endlocal