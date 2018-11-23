-- Cinematic Camera MegaMol Configuration File --

print("I am the MegaMol VISUS CinematicCamera cluster configuration!")

basePath = "\\\\vestastore\\Entwicklung\\braunms\\cinematiccamera\\"

mmSetConfigValue("headNode",   "minyou")
mmSetConfigValue("renderHead", "10.35.1.1")

mmSetLogLevel(     0)
mmSetEchoLevel(    "*")
mmSetAppDir(       basePath .. "bin")
mmAddShaderDir(    basePath .. "share\\shaders")
mmAddResourceDir(  basePath .. "share\\resources")
mmPluginLoaderInfo(basePath .. "bin", "*.mmplg", "include")

mmSetConfigValue("consolegui", "off")
mmSetConfigValue("topmost",    "on")
mmSetConfigValue("vsync",      "off")

computer   = mmGetMachineName()
node_count = 20

--- Load cinematic parameters ---
local cinematic = require("cinematic_params")
mmSetConfigValue("cinematic_width",         tostring(cinematic.width))
mmSetConfigValue("cinematic_height",        tostring(cinematic.height))
mmSetConfigValue("cinematic_fps",           tostring(cinematic.fps))
mmSetConfigValue("cinematic_background",    tostring(cinematic.background))
mmSetConfigValue("cinematic_luaFileToLoad", tostring(cinematic.luaFileToLoad))
mmSetConfigValue("cinematic_keyframeFile",  tostring(cinematic.keyframeFile))


if string.lower(computer) == "minyou" then
    mmSetConfigValue("*-window",   "x5y35w2000h1000")
    mmSetConfigValue("consolegui", "on")
    mmSetConfigValue("topmost",    "off")
    mmSetConfigValue("fullscreen", "off")
else
    keshikinumber = string.match(string.lower(computer), "keshiki(%d+)")
    if keshikinumber ~= nil then
    
        print("I think I am keshiki" .. keshikinumber)

        mmSetConfigValue("scsudptarget", "10.35.1.1")
        mmSetConfigValue("scservername", "10.35.3.1")

        node_index = keshikinumber - 1
               
        -- Setting virtual viewport for tiles
        cc_W_int      = tonumber(tostring(cinematic.width))
        cc_H_int      = tonumber(tostring(cinematic.height))           
        cc_W_str      = tostring(cinematic.width)
        cc_H_str      = tostring(cinematic.height)
        cc_tile_X_str = tostring(cc_W_int // node_count * node_index)
        cc_tile_W_str = tostring(cc_W_int // node_count)
        cc_tile_H_str = cc_H_str

        mmSetConfigValue("t1-window", "x0y0w" .. cc_tile_W_str .. "h" .. cc_tile_H_str .. "nd") 
        mmSetConfigValue("tvview",    cc_W_str .. ";" .. cc_H_str)   
        mmSetConfigValue("t1-tvtile", cc_tile_X_str .. ";0;" .. cc_tile_W_str .. ";" .. cc_tile_H_str)   
        mmSetConfigValue("tvproj",    "mono")
    end
end

