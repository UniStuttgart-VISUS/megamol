## GLStateChecker.py

This utility automatically checks the GL state when a callback of a Renderer[23]DModule is entered and left. Same applies to Create(). To use it, follow these steps:
1. compile MegaMol after uncommenting `#define RIG_RENDERCALLS_WITH_DEBUGGROUPS` in core\include\mmcore\RigRendering.h
2. have Python 3.7 or newer installed
3. have jsondiff installed `pip install jsondiff --user`
4. install apitrace https://apitrace.github.io/ somewhere
5. fix the apitrace path in `GLStateChecker.py` `apitrace = "T:\\Utilities\\apitrace-msvc\\x64\\bin\\apitrace.exe"`
6. go to the MegaMol bin directory
7. run `python <path to this directory>\GLStateChecker.py <your usual mmconsole arguments>`
8. Frames 0 and 1 as well as 1 and 2 will looked at 