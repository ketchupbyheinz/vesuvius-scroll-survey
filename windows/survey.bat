@echo off
rem Usage: survey.bat PHerc0211 [extra options for vesuvius_survey.survey]
rem Traces, renders, inks and scores patches of one scroll; results in work\survey\<scroll>\*\card.png
setlocal
cd /d "%~dp0\.."
set ROOT=%CD%
set PY=%ROOT%\env\Scripts\python.exe
for /r "%ROOT%\apps\VC3D" %%F in (vc_grow_seg_from_seed.exe) do if exist "%%F" set VC=%%F
if "%~1"=="" (echo Usage: survey.bat SCROLL_NAME e.g. PHerc0211 & exit /b 1)
"%PY%" -m vesuvius_survey.survey --scroll %1 --out "%ROOT%\work\survey" --vc "%VC%" --ckpt-dir "%ROOT%\ckpt" %2 %3 %4 %5 %6 %7 %8
pause
