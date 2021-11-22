## GLStateChecker.py

This utility automatically checks the GL state when a callback of a Renderer[23]DModule is entered and left. Same applies to Create(). To use it, follow these steps:
1. compile MegaMol after uncommenting `#define RIG_RENDERCALLS_WITH_DEBUGGROUPS` in core\include\mmcore\RigRendering.h
1. have Python 3.7 or newer installed
1. have jsondiff installed `pip install jsondiff --user`
1. install apitrace https://apitrace.github.io/ somewhere
1. go to the MegaMol install/bin directory (where megamol.exe is)
1. run `python <path to this directory>\GLStateChecker.py --apitrace <path to apitrace.exe> --exe <mmconsole.exe or megamol.exe> -- <your usual arguments>`
1. close MegaMol
1. Frames 0 and 1 as well as 1 and 2 will looked at. This involves a lot of apitrace replace for all rigged calls, so please be patient.
