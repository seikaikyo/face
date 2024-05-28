echo @echo off > run_app.bat
echo call myenv\Scripts\activate >> run_app.bat
echo python app.py >> run_app.bat
echo pause >> run_app.bat