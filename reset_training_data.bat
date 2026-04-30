@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo Resetting training images and trained models...
attrib -R "TrainingImage\*" /S /D >nul 2>&1
attrib -R "TrainingImageLabel\Trainner.yml" >nul 2>&1
attrib -R "TrainingImageLabel\face_model.npz" >nul 2>&1
attrib -R "TrainingImageLabel\arcface_embeddings.npz" >nul 2>&1
del /f /q "TrainingImage\*" >nul 2>&1
del /f /q "TrainingImageLabel\Trainner.yml" >nul 2>&1
del /f /q "TrainingImageLabel\face_model.npz" >nul 2>&1
del /f /q "TrainingImageLabel\arcface_embeddings.npz" >nul 2>&1
echo Done.
echo Student list and attendance files were kept.
pause
