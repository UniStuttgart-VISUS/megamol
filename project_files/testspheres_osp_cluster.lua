aggregate = "true"
headNode = "127.0.0.1"
renderHead = "127.0.0.1"

rank = mmGetEnvValue("PMI_RANK")
role = mmGetConfigValue("role")
print("I am a " .. role)
if role == "head" then

    mmCreateView("testview", "GUIView", "gui")
    mmCreateModule("View3D", "::testview::v")
    mmCreateJob("simpleclusterserver", "SimpleClusterServer", "::scs")
    mmSetParamValue("::scs::viewname", "::testview::v")
    mmSetParamValue("::scs::server::Name", headNode)
    mmSetParamValue("::scs::udptarget", renderHead)
    mmSetParamValue("::scs::server::noEcho", "true")

    mmCreateModule("FBOCompositor2", "::testview::fboc")

    mmCreateCall("CallRenderView", "::testview::gui::renderview", "::testview::v::render")
    mmCreateCall("CallRender3D", "::testview::v::rendering", "::testview::fboc::rendering")
else

    mmCreateModule("View3D", "::testview::v")
    mmCreateModule("OSPRaySphereGeometry", "::testview::geo")
    mmCreateModule("OSPRayOBJMaterial", "::testview::mat")
    mmCreateModule("OSPRayRenderer", "::testview::rnd")
    mmSetParamValue("::testview::rnd::accumulate", "False")
    mmCreateModule("OSPRayAmbientLight", "::testview::light")
    mmCreateModule("TestSpheresDataSource", "::testview::dat")

    mmCreateCall("CallOSPRayMaterial", "::testview::rnd::getMaterialSlot", "::testview::mat::deployMaterialSlot")
    mmCreateCall("CallOSPRayStructure", "::testview::rnd::getStructure", "::testview::geo::deployStructureSlot")
    mmCreateCall("CallOSPRayLight", "::testview::rnd::getLight", "::testview::light::deployLightSlot")
    mmCreateCall("CallRender3D", "::testview::v::rendering", "::testview::rnd::rendering")
    mmCreateCall("MultiParticleDataCall", "::testview::geo::getdata", "::testview::dat::getData")

    mmCreateModule("FBOTransmitter2", "::testview::fbot")
    mmSetParamValue("::testview::fbot::view", "::testview::v")
    mmSetParamValue("::testview::fbot::aggregate", aggregate)
    
    mmSetParamValue("::testview::fbot::port", tostring(37230 + rank))
    mmSetParamValue("::testview::fbot::targetMachine", headNode)
    mmSetParamValue("::testview::fbot::trigger", " ")

    if headNode == "127.0.0.1" and renderHead == "127.0.0.1" then
        mmSetParamValue("::testview::fbot::force_localhost", "True")
    end

    mmCreateCall("MpiCall", "::testview::fbot::requestMpi", "::testview::mpi::provideMpi")
end
