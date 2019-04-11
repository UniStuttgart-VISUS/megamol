mmpldFile = "\\\\vestastore\\Demos\\megamol-data\\sfb716-20140605\\b5\\B5.vid1.mmpld"
siffFile = "\\\\vestastore\\Demos\\megamol-data\\sfb716-20140605\\b5\\time2b.siff"
headNode = mmGetConfigValue("headNode")
renderHead = mmGetConfigValue("renderHead")
showSplines = "false"
machine = mmGetMachineName()
role = mmGetConfigValue("role")

print("I am a " .. role .. " running on " .. machine)

function doRendering(spl, sfile, mfile)
  if spl == "true" then
    mmSetParamValue("::b5_demo::v::backCol", "black")
    mmSetParamValue("::b5_demo::v::showBBox", "false")
    mmSetParamValue("::b5_demo::v::stereo::eyeDist", "0.1")
    mmSetParamValue("::b5_demo::v::viewcube::show", "false")
    mmCreateModule("BezierCPUMeshRenderer", "::b5_demo::splines")
    mmCreateModule("SIFFDataSource", "::b5_demo::data")
    mmSetParamValue("::b5_demo::data::filename", sfile)
    mmCreateModule("SiffCSplineFitter", "::b5_demo::splineFitter")
    mmSetParamValue("::b5_demo::splineFitter::deCycle", "true")
    mmCreateCall("CallRender3D", "::b5_demo::v::rendering", "::b5_demo::splines::rendering")
    mmCreateCall("BezierCurvesListDataCall", "::b5_demo::splines::getdata", "::b5_demo::splineFitter::getdata")
    mmCreateCall("MultiParticleDataCall", "::b5_demo::splineFitter::indata", "::b5_demo::data::getdata")

  else
    mmSetParamValue("::b5_demo::v::backCol", "black")
    mmSetParamValue("::b5_demo::v::showBBox", "false")
    mmSetParamValue("::b5_demo::v::stereo::eyeDist", "0.05")
    mmSetParamValue("::b5_demo::v::viewcube::show", "false")
    mmCreateModule("SimpleSphereRenderer", "::b5_demo::rnd")
    mmSetParamValue("::b5_demo::rnd::renderMode", "Simple") 
    mmCreateModule("MMPLDDataSource", "::b5_demo::dat")
    mmSetParamValue("::b5_demo::dat::filename", mfile)
    mmCreateModule("LinearTransferFunction", "::b5_demo::tf")
    mmSetParamValue("::b5_demo::tf::mincolour", "#22aaff")
    mmSetParamValue("::b5_demo::tf::enable01", "false")
    mmSetParamValue("::b5_demo::tf::colour01", "#0000d3")
    mmSetParamValue("::b5_demo::tf::enable02", "false")
    mmSetParamValue("::b5_demo::tf::colour02", "#002aff")
    mmSetParamValue("::b5_demo::tf::enable03", "false")
    mmSetParamValue("::b5_demo::tf::colour03", "#0080ff")
    mmSetParamValue("::b5_demo::tf::enable04", "false")
    mmSetParamValue("::b5_demo::tf::colour04", "#00d4ff")
    mmSetParamValue("::b5_demo::tf::enable05", "false")
    mmSetParamValue("::b5_demo::tf::colour05", "#2bffd4")
    mmSetParamValue("::b5_demo::tf::enable06", "false")
    mmSetParamValue("::b5_demo::tf::colour06", "#81ff7e")
    mmSetParamValue("::b5_demo::tf::enable07", "false")
    mmSetParamValue("::b5_demo::tf::colour07", "#d4ff2b")
    mmSetParamValue("::b5_demo::tf::enable08", "false")
    mmSetParamValue("::b5_demo::tf::colour08", "#ffd400")
    mmSetParamValue("::b5_demo::tf::enable09", "false")
    mmSetParamValue("::b5_demo::tf::colour09", "#ff7e00")
    mmSetParamValue("::b5_demo::tf::enable10", "false")
    mmSetParamValue("::b5_demo::tf::colour10", "#ff2b00")
    mmSetParamValue("::b5_demo::tf::enable11", "false")
    mmSetParamValue("::b5_demo::tf::colour11", "#db0000")
    mmSetParamValue("::b5_demo::tf::maxcolour", "coral")
    mmCreateCall("CallRender3D", "::b5_demo::v::rendering", "::b5_demo::rnd::rendering")
    mmCreateCall("MultiParticleDataCall", "::b5_demo::rnd::getdata", "::b5_demo::dat::getdata")
    mmCreateCall("CallGetTransferFunction", "::b5_demo::rnd::gettransferfunction", "::b5_demo::tf::gettransferfunction")

  end
end


if role == "head" then
  mmCreateView("b5_demo", "View3DSpaceMouse", "v")
  
  mmCreateJob("simpleclusterserver", "SimpleClusterServer", "::scs")
  mmSetParamValue("::scs::viewname", "::b5_demo::v")
  mmSetParamValue("::scs::server::Name", headNode)
  mmSetParamValue("::scs::udptarget", renderHead)
  mmSetParamValue("::scs::server::noEcho", "true")

  doRendering(showSplines, siffFile, mmpldFile)

else

  mmCreateModule("View3D", "::b5_demo::v")

  doRendering(showSplines, siffFile, mmpldFile)

end