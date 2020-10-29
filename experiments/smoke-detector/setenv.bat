@echo off

if [%GOVTECH_PATH%] == [] (set GOVTECH_PATH=../..)

set PYTHONPATH=%GOVTECH_PATH%
set CASE_DETECTOR_PATH=%GOVTECH_PATH%/experiments/govtech-case-detector
set ITEM_DETECTOR_PATH=%GOVTECH_PATH%/experiments/govtech