pdbFile = "\\\\vestastore\\Demos\\megamol-data\\ccmv_morph_movie_06102010-2.pdb"
xtcFile = "\\\\vestastore\\Demos\\megamol-data\\ccmv_morph_movie_06102010-2.xtc"
headNode = mmGetConfigValue("headNode")
renderHead = mmGetConfigValue("renderHead")
machine = mmGetMachineName()
role = mmGetConfigValue("role")

print("I am a " .. role .. " running on " .. machine)

if role == "head" then
  
  mmCreateView("mdao_demo", "View3DSpaceMouse", "v")
  
  mmCreateJob("simpleclusterserver", "SimpleClusterServer", "::scs")
  mmSetParamValue("::scs::viewname", "::mdao_demo::v")
  mmSetParamValue("::scs::server::Name", headNode)
  mmSetParamValue("::scs::udptarget", renderHead)
  mmSetParamValue("::scs::server::noEcho", "true")
  
  mmCreateModule("PDBLoader", "::mdao_demo::PDBLoader1")
  mmSetParamValue("::mdao_demo::PDBLoader1::pdbFilename", pdbFile)
  mmSetParamValue("::mdao_demo::PDBLoader1::xtcFilename", xtcFile)
  mmSetParamValue("::mdao_demo::PDBLoader1::strideFlag", "False")
  mmCreateModule("AOSphereRenderer", "::mdao_demo::AOSphereRenderer1")
  mmSetParamValue("::mdao_demo::AOSphereRenderer1::vol::sizex", "16")
  mmSetParamValue("::mdao_demo::AOSphereRenderer1::vol::sizey", "16")
  mmSetParamValue("::mdao_demo::AOSphereRenderer1::vol::sizez", "16")
  mmSetParamValue("::mdao_demo::AOSphereRenderer1::ao::acc", "1")
  mmSetParamValue("::mdao_demo::AOSphereRenderer1::ao::evalFac", "1.5")
  mmSetParamValue("::mdao_demo::v::backCol", "black")
  mmSetParamValue("::mdao_demo::v::showBBox", "False")
  mmSetParamValue("::mdao_demo::v::stereo::eyeDist", "0.1")
  mmSetParamValue("::mdao_demo::v::viewcube::show", "False")
  mmSetParamValue("::mdao_demo::v::anim::speed", "10")
  mmCreateModule("LinearTransferFunction", "::mdao_demo::LinearTransferFunction1")
  mmSetParamValue("::mdao_demo::LinearTransferFunction1::mincolour", "lightblue")
  mmSetParamValue("::mdao_demo::LinearTransferFunction1::maxcolour", "lightblue")
  mmCreateCall("CallRender3D", "::mdao_demo::v::rendering", "::mdao_demo::AOSphereRenderer1::rendering")
  mmCreateCall("CallGetTransferFunction", "::mdao_demo::AOSphereRenderer1::gettransferfunction", "::mdao_demo::LinearTransferFunction1::gettransferfunction")
  mmCreateCall("MolecularDataCall", "::mdao_demo::AOSphereRenderer1::getdata", "::mdao_demo::PDBLoader1::dataout")

else
  mmCreateModule("View3D", "::mdao_demo::v")
  --mmCreateView("mdao_demo", "View3D", "v")

  mmCreateModule("PDBLoader", "::mdao_demo::PDBLoader1")
  mmSetParamValue("::mdao_demo::PDBLoader1::pdbFilename", pdbFile)
  mmSetParamValue("::mdao_demo::PDBLoader1::xtcFilename", xtcFile)
  mmSetParamValue("::mdao_demo::PDBLoader1::strideFlag", "False")
  mmCreateModule("AOSphereRenderer", "::mdao_demo::AOSphereRenderer1")
  mmSetParamValue("::mdao_demo::AOSphereRenderer1::vol::sizex", "16")
  mmSetParamValue("::mdao_demo::AOSphereRenderer1::vol::sizey", "16")
  mmSetParamValue("::mdao_demo::AOSphereRenderer1::vol::sizez", "16")
  mmSetParamValue("::mdao_demo::AOSphereRenderer1::ao::acc", "1")
  mmSetParamValue("::mdao_demo::AOSphereRenderer1::ao::evalFac", "1.5")
  mmSetParamValue("::mdao_demo::v::backCol", "black")
  mmSetParamValue("::mdao_demo::v::showBBox", "False")
  mmSetParamValue("::mdao_demo::v::stereo::eyeDist", "0.1")
  mmSetParamValue("::mdao_demo::v::viewcube::show", "False")
  mmSetParamValue("::mdao_demo::v::anim::speed", "10")
  mmCreateModule("LinearTransferFunction", "::mdao_demo::LinearTransferFunction1")
  mmSetParamValue("::mdao_demo::LinearTransferFunction1::mincolour", "lightblue")
  mmSetParamValue("::mdao_demo::LinearTransferFunction1::maxcolour", "lightblue")
  mmCreateCall("CallRender3D", "::mdao_demo::v::rendering", "::mdao_demo::AOSphereRenderer1::rendering")
  mmCreateCall("CallGetTransferFunction", "::mdao_demo::AOSphereRenderer1::gettransferfunction", "::mdao_demo::LinearTransferFunction1::gettransferfunction")
  mmCreateCall("MolecularDataCall", "::mdao_demo::AOSphereRenderer1::getdata", "::mdao_demo::PDBLoader1::dataout")


end