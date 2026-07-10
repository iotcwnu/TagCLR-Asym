@echo off
chcp 65001 >nul
setlocal

:: 切换到批处理文件所在目录
cd /d "%~dp0"

echo ==================================================
echo 即将删除以下目录及所有子目录中的 MP4 文件：
echo %cd%
echo.
echo TXT 文件不会被删除。
echo 注意：删除后不会进入回收站！
echo ==================================================
echo.

:: 先列出将要删除的 MP4 文件
for /r %%F in (*.mp4) do (
    echo %%F
)

echo.
choice /c YN /n /m "确认删除以上所有 MP4 文件吗？[Y/N]："

:: 选择 N 时取消
if errorlevel 2 goto cancel

echo.
echo 正在删除 MP4 文件...

set /a count=0

:: 递归删除当前目录及所有子目录中的 MP4 文件
for /r %%F in (*.mp4) do (
    del /f /q "%%~fF"
    if not exist "%%~fF" set /a count+=1
)

echo.
echo 删除完成，共删除 %count% 个 MP4 文件。
echo TXT 文件已保留。
pause
exit

:cancel
echo.
echo 已取消，没有删除任何文件。
pause
