@echo off
set gpu_n=%1
set DATASET=%2

set seed=5
set BATCH_SIZE=32
set SLIDE_WIN=5
set dim=64
set out_layer_num=1
set SLIDE_STRIDE=1
set topk=5
set out_layer_inter_dim=128
set val_ratio=0.2
set decay=0

set path_pattern=%DATASET%
set COMMENT=%DATASET%

set EPOCH=30
set report=best

if "%gpu_n%"=="cpu" (
    python main.py ^
        -dataset %DATASET% ^
        -save_path_pattern %path_pattern% ^
        -slide_stride %SLIDE_STRIDE% ^
        -slide_win %SLIDE_WIN% ^
        -batch %BATCH_SIZE% ^
        -epoch %EPOCH% ^
        -comment %COMMENT% ^
        -random_seed %seed% ^
        -decay %decay% ^
        -dim %dim% ^
        -out_layer_num %out_layer_num% ^
        -out_layer_inter_dim %out_layer_inter_dim% ^
        -decay %decay% ^
        -val_ratio %val_ratio% ^
        -report %report% ^
        -topk %topk% ^
        -device "cpu"
) else (
    set CUDA_VISIBLE_DEVICES=%gpu_n%
    python main.py ^
        -dataset %DATASET% ^
        -save_path_pattern %path_pattern% ^
        -slide_stride %SLIDE_STRIDE% ^
        -slide_win %SLIDE_WIN% ^
        -batch %BATCH_SIZE% ^
        -epoch %EPOCH% ^
        -comment %COMMENT% ^
        -random_seed %seed% ^
        -decay %decay% ^
        -dim %dim% ^
        -out_layer_num %out_layer_num% ^
        -out_layer_inter_dim %out_layer_inter_dim% ^
        -decay %decay% ^
        -val_ratio %val_ratio% ^
        -report %report% ^
        -topk %topk%
)
