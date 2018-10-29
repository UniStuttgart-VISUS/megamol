-- MegaMol Configuration File --

print("I am the MegaMol VISUS CinematicCamera cluster configuration!")

basePath = "u:\\home\\reina\\src\\megamol-dev\\"
basePath = 

mmSetLogLevel(0)
mmSetEchoLevel("*")
mmSetAppDir(basePath .. "bin")
mmAddShaderDir(basePath .. "share\\shaders")
mmAddResourceDir(basePath .. "share\\resources")
mmPluginLoaderInfo(basePath .. "bin", "*.mmplg", "include")

computer = mmGetMachineName()

mmSetConfigValue("*-window",   "x5y35w1280h720")
mmSetConfigValue("consolegui", "on")
mmSetConfigValue("topmost",    "off")
mmSetConfigValue("fullscreen", "off")