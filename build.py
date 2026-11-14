#!/usr/bin/env python3
"""Rebuild local experiment from the installed pinned app and model."""
import pathlib,subprocess,shutil,sys
root=pathlib.Path(__file__).resolve().parent
app=pathlib.Path('/Library/Input Methods/DoubaoIme.app/Contents')
def run(*args):subprocess.run(list(map(str,args)),cwd=root,check=True)
run(sys.executable,root/'prepare.py')
if not (root/'Frameworks').exists():shutil.copytree(app/'Frameworks',root/'Frameworks',symlinks=True)
run('clang','-dynamiclib',root/'helper.c','-o',root/'helper.dylib')
run('clang','-dynamiclib',root/'helper.c','-o',root/'Frameworks/libDoubaoImeDylib.dylib','-Wl,-install_name,@rpath/libDoubaoImeDylib.dylib')
run('codesign','-f','-s','-',root/'runtime.bundle')
run('clang++','-std=c++17','-framework','Foundation',root/'run.mm','-o',root/'run','-Wl,-rpath,@loader_path/Frameworks')
run(sys.executable,root/'repack.py')
home=pathlib.Path.home()
(root/'offline.sb').write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* file-write* (subpath "/Library/Input Methods/DoubaoIme.app"))\n(deny file-read* file-write* (subpath '+__import__('json').dumps(str(home/'Library/Application Support/DoubaoIme'))+'))\n')
print('Built. Run: python3 transcribe.py test.wav')
