file = "\\\\vestastore\\Demos\\HR-Bilder\\Mono\\Handgekloeppelt\\Grossgeraeteantrag.png"
headNode = mmGetConfigValue("headNode")
renderHead = mmGetConfigValue("renderHead")
machine = mmGetMachineName()
role = mmGetConfigValue("role")

print("I am a " .. role .. " running on " .. machine)

if role == "head" then
  
  mmCreateView("img_demo", "View3DSpaceMouse", "v")

  mmCreateJob("simpleclusterserver", "SimpleClusterServer", "::scs")
  mmSetParamValue("::scs::viewname", "::img_demo::v")
  mmSetParamValue("::scs::server::Name", headNode)
  mmSetParamValue("::scs::udptarget", renderHead)
  mmSetParamValue("::scs::server::noEcho", "true")

  mmSetParamValue("::img_demo::v::backCol", "black")
  mmSetParamValue("::img_demo::v::showBBox", "off")
  mmSetParamValue("::img_demo::v::viewKey::MoveStep", "100")
  mmSetParamValue("::img_demo::v::viewcube::show", "false")
  mmSetParamValue("::img_demo::v::resetViewOnBBoxChange", "true")
  mmCreateModule("ImageViewer", "::img_demo::image")
  mmSetParamValue("::img_demo::image::leftImg", file)
  mmCreateCall("CallRender3D", "::img_demo::v::rendering", "::img_demo::image::rendering")

else

  mmCreateModule("View3D", "::img_demo::v")
  mmSetParamValue("::img_demo::v::backCol", "black")
  mmSetParamValue("::img_demo::v::showBBox", "off")
  mmSetParamValue("::img_demo::v::viewKey::MoveStep", "100")
  mmSetParamValue("::img_demo::v::viewcube::show", "false")
  mmSetParamValue("::img_demo::v::resetViewOnBBoxChange", "true")
  mmCreateModule("ImageViewer", "::img_demo::image")
  mmSetParamValue("::img_demo::image::leftImg", file)
  mmCreateCall("CallRender3D", "::img_demo::v::rendering", "::img_demo::image::rendering")

end
