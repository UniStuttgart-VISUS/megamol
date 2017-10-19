print('Hi, I am the megamolconfig.lua!')

-- mmSetAppDir("U:/home/reina/src/megamol-dev/bin")
mmSetAppDir(".")

mmSetLogFile("")
mmSetLogLevel(0)
mmSetEchoLevel('*')

mmAddShaderDir("U:/home/reina/src/megamol-dev/share/shaders")
mmAddResourceDir("U:/home/reina/src/megamol-dev/share/resources")

mmPluginLoaderInfo("U:/home/reina/src/megamol-dev/bin", "*.mmplg", "include")

-- mmSetConfigValue("*-window", "w1280h720")
mmSetConfigValue("*-window", "w720h720")
mmSetConfigValue("consolegui", "on")

mmSetConfigValue("LRHostEnable", "true")

return "done with megamolconfig.lua."
-- error("megamolconfig.lua is not happy!")