chkptFile = "\\\\vestastore\\demos\\megamol-data\\huebsch.chkpt"
headNode = mmGetConfigValue("headNode")
renderHead = mmGetConfigValue("renderHead")
machine = mmGetMachineName()
role = mmGetConfigValue("role")
useGrim = "false"

print("I am a " .. role .. " running on " .. machine)

function doRendering(grim, cfile)

  mmCreateModule("IMDAtomData", "::imd_demo::data")
  mmSetParamValue("::imd_demo::data::filename", chkptFile)
  mmSetParamValue("::imd_demo::data::radius", 2)
  mmSetParamValue("::imd_demo::data::colcolumn", "epot")
  mmSetParamValue("::imd_demo::data::dir::colcolumn", "epot")

  mmCreateModule("LinearTransferFunction", "::imd_demo::tf")
  mmSetParamValue("::imd_demo::tf::mincolour", "slategray")
  mmSetParamValue("::imd_demo::tf::maxcolour", "yellow")
  mmSetParamValue("::imd_demo::tf::enable03", "true")
  mmSetParamValue("::imd_demo::tf::colour03", "slategray")
  mmSetParamValue("::imd_demo::tf::value03", 0.125000)
  mmSetParamValue("::imd_demo::tf::enable06", "true")
  mmSetParamValue("::imd_demo::tf::colour06", "red")
  mmSetParamValue("::imd_demo::tf::value06", 0.500000)


  if useGrim == "true" then

    mmCreateModule("GrimRenderer", "::imd_demo::spheres")
    mmCreateModule("DataGridder", "::imd_demo::grid")
    mmCreateCall("CallRender3D", "::imd_demo::v::rendering", "::imd_demo::spheres::rendering")
    mmCreateCall("CallGetTransferFunction", "::imd_demo::spheres::gettransferfunction", "::imd_demo::tf::gettransferfunction")
    mmCreateCall("MultiParticleDataCall", "::imd_demo::grid::indata", "::imd_demo::data::getdata")
    mmCreateCall("ParticleGridDataCall", "::imd_demo::spheres::getdata", "::imd_demo::grid::outdata")

  else

    mmCreateModule("SimpleSphereRenderer", "::imd_demo::spheres")
    mmSetParamValue("::imd_demo::spheres::renderMode", "Simple") 
    mmCreateModule("AddParticleColours", "::imd_demo::colourizer")
    mmCreateModule("ClipPlane", "::imd_demo::clip")
    mmCreateCall("CallRender3D", "::imd_demo::v::rendering", "::imd_demo::spheres::rendering")
    mmCreateCall("MultiParticleDataCall", "::imd_demo::spheres::getdata", "::imd_demo::colourizer::putdata")
    mmCreateCall("MultiParticleDataCall", "::imd_demo::colourizer::getdata", "::imd_demo::data::getdata")
    mmCreateCall("CallGetTransferFunction", "::imd_demo::colourizer::gettransferfunction", "::imd_demo::tf::gettransferfunction")
    mmCreateCall("CallClipPlane", "::imd_demo::spheres::getclipplane", "::imd_demo::clip::getclipplane")
      
  end
end

if role == "head" then
  
  mmCreateView("imd_demo", "View3DSpaceMouse", "v")
  mmSetParamValue("::imd_demo::v::showBBox", "False")
  mmSetParamValue("::imd_demo::v::viewcube::show", "False")

  mmCreateJob("simpleclusterserver", "SimpleClusterServer", "::scs")
  mmSetParamValue("::scs::viewname", "::imd_demo::v")
  mmSetParamValue("::scs::server::Name", headNode)
  mmSetParamValue("::scs::udptarget", renderHead)
  mmSetParamValue("::scs::server::noEcho", "true")

  doRendering(useGrim, chkptFile)

else
  mmCreateModule("View3D", "::imd_demo::v")
  mmSetParamValue("::imd_demo::v::showBBox", "False")
  mmSetParamValue("::imd_demo::v::viewcube::show", "False")

  doRendering(useGrim, chkptFile)

end