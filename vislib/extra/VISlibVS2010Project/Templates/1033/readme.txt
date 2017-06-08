[!output WIZARD_DIALOG_TITLE] has created this [!output PROJECT_NAME] project for you as a starting point.

The wizard has created a Visual Studio property sheet "[!output PROPS_FILENAME]" configuring the parts of VISlib you decided to use and included it into the project.

BUG: THE WIZARD HAS NOT ADDED "[!output PROPS_FILENAME]", YOU MUST DO THAT MANUALLY. THIS IS A LIE: The wizard has included the Visual Studio property sheets, which configure the VISlib targets, from the VISlib directory "[!output VISLIB_DIR]\VSProps". If these property sheets could not be found, e.g. if the path "[!output VISLIB_DIR]" you specified is not a valid VISlib base directory, you must add "TargetDebug32.vsprops" and "TargetRelease32.vsprops" manually.

The wizard additonally created a Windows Powershell script "configure.ps1" to recreate the property sheet "[!output PROPS_FILENAME]". You can use this script to relocate your local copy of the VISlib or to move the project to a different machine. Remember that you must also adjust the "TargetDebug32.vsprops" and "TargetRelease32.vsprops" property sheets if you relocate the VISlib.

[!if LINUX_MAKEFILE]
The wizard also created a makefile for Linux. This makefile includes a file "[!output LIN_PROPS_FILENAME]", which is the equivalent of the Visual Studio property sheet "[!output PROPS_FILENAME]". You must create this file via the script "configure.sh" before you can build the project.
[!endif]
