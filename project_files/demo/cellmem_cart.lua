pdbFile = "\\\\vestastore\\demos\\megamol-data\\C4.SMD_trajectory.pdb"
xtcFile = "\\\\vestastore\\demos\\megamol-data\\C4.SMD_trajectory.xtc"

headNode = mmGetConfigValue("headNode")
renderHead = mmGetConfigValue("renderHead")
machine = mmGetMachineName()
role = mmGetConfigValue("role")
renderer = "cartoon"

print("I am a " .. role .. " running on " .. machine)

function doRendering(id, pfile, xfile)

    mmCreateModule("PDBLoader", "::cellmem_demo::pdbdata")
    mmSetParamValue("::cellmem_demo::pdbdata::pdbFilename", pfile)
    mmSetParamValue("::cellmem_demo::pdbdata::xtcFilename", xfile)

    if id == "molecule" then

        mmCreateModule("SimpleMoleculeRenderer", "::cellmem_demo::molren")
        mmCreateCall("CallRender3D", "::cellmem_demo::v::rendering", "::cellmem_demo::molren::rendering")
        mmCreateCall("MolecularDataCall", "::cellmem_demo::molren::getData", "::cellmem_demo::::pdbdata::dataout")

    elseif id == "cartoon" then

        mmCreateModule("MoleculeCartoonRenderer", "::cellmem_demo::cartoonren")
        mmSetParamValue("::cellmem_demo::cartoonren::renderingMode", "Cartoon Hybrid")
        mmCreateCall("CallRender3D", "::cellmem_demo::v::rendering", "::cellmem_demo::cartoonren::rendering")
        mmCreateCall("MolecularDataCall", "::cellmem_demo::cartoonren::getdata", "::cellmem_demo::pdbdata::dataout")
    
    elseif id == "molecule+cartoon" then

        mmCreateModule("MoleculeCartoonRenderer", "::cellmem_demo::cartoonren")
        mmCreateModule("SimpleMoleculeRenderer", "::cellmem_demo::molren")
        mmCreateCall("CallRender3D", "::cellmem_demo::v::rendering", "::cellmem_demo::cartoonren::rendering")
        mmCreateCall("CallRender3D", "::cellmem_demo::cartoonren::renderMolecule", "::cellmem_demo::molren::rendering")
        mmCreateCall("MolecularDataCall", "::cellmem_demo::cartoonren::getdata", "::cellmem_demo::pdbdata::dataout")
        mmCreateCall("MolecularDataCall", "::cellmem_demo::molren::getData", "::cellmem_demo::pdbdata::dataout")
        
    elseif id == "molecule+volume" then

        mmCreateModule("ProteinVolumeRenderer", "::cellmem_demo::volren")
        mmCreateModule("SimpleMoleculeRenderer", "::cellmem_demo::molren")
        mmCreateCall("CallRender3D", "::cellmem_demo::v::rendering", "::cellmem_demo::volren::rendering")
        mmCreateCall("CallRender3D", "::cellmem_demo::volren::renderProtein", "::cellmem_demo::molren::rendering")
        mmCreateCall("MolecularDataCall", "::cellmem_demo::volren::getData", "::cellmem_demo::pdbdata::dataout")
        mmCreateCall("MolecularDataCall", "::cellmem_demo::molren::getData", "::cellmem_demo::pdbdata::dataout")
        
    elseif id == "ses" then

        mmCreateModule("MoleculeSESRenderer", "::cellmem_demo::sesren")
        mmCreateCall("CallRender3D", "::cellmem_demo::v::rendering", "::cellmem_demo::sesren::rendering")
        mmCreateCall("MolecularDataCall", "::cellmem_demo::sesren::getData", "::cellmem_demo::pdbdata::dataout")

    end
end


if role == "head" then
  
    mmCreateView("cellmem_demo", "View3DSpaceMouse", "v")
    mmSetParamValue("::cellmem_demo::v::showBBox", "False")
    mmSetParamValue("::cellmem_demo::v::viewcube::show", "False")
  
    mmCreateJob("simpleclusterserver", "SimpleClusterServer", "::scs")
    mmSetParamValue("::scs::viewname", "::cellmem_demo::v")
    mmSetParamValue("::scs::server::Name", headNode)
    mmSetParamValue("::scs::udptarget", renderHead)
    mmSetParamValue("::scs::server::noEcho", "true")
  
    doRendering(renderer, pdbFile, xtcFile)
  
  else

    mmCreateModule("View3D", "::cellmem_demo::v")
    mmSetParamValue("::cellmem_demo::v::showBBox", "False")
    mmSetParamValue("::cellmem_demo::v::viewcube::show", "False")
  
    doRendering(renderer, pdbFile, xtcFile)

  end