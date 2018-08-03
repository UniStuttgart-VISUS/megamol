
crystFile = "\\\\vestastore\\Demos\\megamol-data\\Q_param.txt"
posFile = "\\\\vestastore\\Demos\\megamol-data\\objects.dat"
attribFile = "\\\\vestastore\\Demos\\megamol-data\\objdfns.dat"
headNode = mmGetConfigValue("headNode")
renderHead = mmGetConfigValue("renderHead")
machine = mmGetMachineName()
role = mmGetConfigValue("role")
renderer = "1"

print("I am a " .. role .. " running on " .. machine)

function doRendering(id, cfile, pfile, afile)


    mmCreateModule("QuartzParticleFortLoader", "::quartz_demo::dat")
    mmSetParamValue("::quartz_demo::dat::positionFile", pfile)
    mmSetParamValue("::quartz_demo::dat::attributeFile", afile)

    mmCreateModule("QuartzCrystalDataSource", "::quartz_demo::cryst")
    mmSetParamValue("::quartz_demo::cryst::filename", cfile)

    mmCreateModule("QuartzDataGridder", "::quartz_demo::grid")
    mmSetParamValue("::quartz_demo::grid::gridsizex", "7")
    mmSetParamValue("::quartz_demo::grid::gridsizey", "7")
    mmSetParamValue("::quartz_demo::grid::gridsizez", "7")

    mmCreateModule("ClipPlane", "::quartz_demo::clip")
    mmSetParamValue("::quartz_demo::clip::enable", "on")
    mmSetParamValue("::quartz_demo::clip::colour", "orange")
    mmSetParamValue("::quartz_demo::clip::normal", "1.0;0.0;0.0")
    mmSetParamValue("::quartz_demo::clip::point", "165.0;165.0;165.0")

    mmCreateCall("QuartzParticleDataCall", "::quartz_demo::grid::datain", "::quartz_demo::dat::dataout")
    mmCreateCall("QuartzCrystalDataCall", "::quartz_demo::grid::crysin", "::quartz_demo::cryst::dataout")


    if id == "1" then

        mmCreateModule("QuartzCrystalRenderer", "::quartz_demo::rnd")
        mmSetParamValue("::quartz_demo::rnd::idx", "1")
        mmCreateCall("CallRender3D", "::quartz_demo::v::rendering", "::quartz_demo::rnd::rendering")
        mmCreateCall("QuartzCrystalDataCall", "::quartz_demo::rnd::datain", "::quartz_demo::cryst::dataout")
    
    elseif id == "2" then
    
        mmSetParamValue("::quartz_demo::v::backCol", "slategray")
        --mmSetParamValue("::quartz_demo::v::camsettings", "{ApertureAngle=30\\nCoordSystemType=2\\nNearClip=1.6520425081253052\\nFarClip=2.8979587554931641\\nProjection=0\\nStereoDisparity=0.30000001192092896\\nStereoEye=0\\nFocalDistance=2.2765483856201172\\nPositionX=-1.8369302749633789\\nPositionY=0.43051132559776306\\nPositionZ=0.32765769958496094\\nLookAtX=0.43698525428771973\\nLookAtY=0.43703389167785645\\nLookAtZ=0.43691778182983398\\nUpX=0.0032014620956033468\\nUpY=-0.99997085332870483\\nUpZ=-0.0069328020326793194\\nTileLeft=0\\nTileBottom=0\\nTileRight=1600\\nTileTop=1138\\nVirtualViewWidth=1600\\nVirtualViewHeight=1138\\n}")
        mmCreateModule("SwitchRenderer3D", "::quartz_demo::switch")
        mmSetParamValue("::quartz_demo::switch::selection", "2")
        mmCreateModule("QuartzRenderer", "::quartz_demo::rnd1")
        mmCreateModule("QuartzTexRenderer", "::quartz_demo::rnd2")
        mmCreateCall("CallRender3D", "::quartz_demo::v::rendering", "::quartz_demo::switch::rendering")
        mmCreateCall("CallRender3D", "::quartz_demo::switch::renderer1", "::quartz_demo::rnd1::rendering")
        mmCreateCall("CallRender3D", "::quartz_demo::switch::renderer2", "::quartz_demo::rnd2::rendering")
        mmCreateCall("QuartzParticleGridDataCall", "::quartz_demo::rnd1::datain", "::quartz_demo::grid::dataout")
        mmCreateCall("QuartzParticleGridDataCall", "::quartz_demo::rnd2::datain", "::quartz_demo::grid::dataout")
        mmCreateCall("QuartzCrystalDataCall", "::quartz_demo::rnd1::typesin", "::quartz_demo::cryst::dataout")
        mmCreateCall("QuartzCrystalDataCall", "::quartz_demo::rnd2::typesin", "::quartz_demo::cryst::dataout")
        mmCreateCall("CallClipPlane", "::quartz_demo::rnd1::clipplane", "::quartz_demo::clip::getclipplane")
        mmCreateCall("CallClipPlane", "::quartz_demo::rnd2::clipplane", "::quartz_demo::clip::getclipplane")
        
        
    elseif id == "3" then

        mmSetParamValue("::quartz_demo::v::backCol", "black")
        mmCreateModule("QuartzPlaneTexRenderer", "::quartz_demo::r")
        mmCreateCall("CallRender2D", "::quartz_demo::v::rendering", "::quartz_demo::r::rendering")
        mmCreateCall("QuartzParticleGridDataCall", "::quartz_demo::r::datain", "::quartz_demo::grid::dataout")
        mmCreateCall("QuartzCrystalDataCall", "::quartz_demo::r::typesin", "::quartz_demo::cryst::dataout")
        mmCreateCall("CallClipPlane", "::quartz_demo::r::clipplane", "::quartz_demo::clip::getclipplane")
        
    elseif id == "4" then
    
        mmSetParamValue("::quartz_demo::v::backCol", "slategray")
        mmCreateModule("PoreNetExtractor", "::quartz_demo::e")
        mmCreateCall("CallRender3D", "::quartz_demo::v::rendering", "::quartz_demo::e::rendering")
        mmCreateCall("QuartzParticleGridDataCall", "::quartz_demo::e::datain", "::quartz_demo::grid::dataout")
        mmCreateCall("QuartzCrystalDataCall", "::quartz_demo::e::typesin", "::quartz_demo::cryst::dataout")
    
    else
        mmLog(1, "No renderer specified")
    end
end



if role == "head" then

    if renderer == "3" then 
        mmCreateView("quartz_demo", "View2D", "v")
    else 
        mmCreateView("quartz_demo", "View3DSpaceMouse", "v")
    end

    mmCreateJob("simpleclusterserver", "SimpleClusterServer", "::scs")
    mmSetParamValue("::scs::viewname", "::quartz_demo::v")
    mmSetParamValue("::scs::server::Name", headNode)
    mmSetParamValue("::scs::udptarget", renderHead)
    mmSetParamValue("::scs::server::noEcho", "true")

    doRendering(renderer, crystFile, posFile, attribFile)

else

    if renderer == "3" then 
        mmCreateModule("View2D", "::quartz_demo::v")
    else 
        mmCreateModule("View3D", "::quartz_demo::v")
    end

    doRendering(renderer, crystFile, posFile, attribFile)

end
