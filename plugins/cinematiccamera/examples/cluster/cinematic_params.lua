-- Cinematic Camera Parameters --

local cinematic = 
{
    width         = 1920                                         , -- Cinematic Viewport Width    (must be > 0)
    height        = 1080                                         , -- Cinematic Viewport Height   (must be > 0)
    fps           = 24                                           , -- Cinematic Frames Per Second (must be > 0)
    background    = 'white'                                      , -- The background color.
    keyframeFile  = "cinematic_keyframes.kf"                     , -- The keyframe file to load.
    luaFileToLoad = "ospray.lua"                                 , -- The lua project file to load.    
}
return cinematic