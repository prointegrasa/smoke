
setlocal

call "run_clean"
call "run_create_train_dataset"
call "run_create_image_lists"
call "run_augment_by_transformations"
call "run_create_image_lists"
call "run_train_val_split"
call "run_detect_anchor_boxes"


endlocal




