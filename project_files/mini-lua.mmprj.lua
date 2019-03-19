role = mmGetConfigValue("role")
print("I am a " .. role)
if role == "head" then
    mmCreateView("mini-lua", "View3D", "v")
    mmCreateJob("simpleclusterserver", "SimpleClusterServer", "::scs")
    mmSetParamValue("::scs::viewname", "::mini-lua::v")
    mmSetParamValue("::scs::server::Name", "127.0.0.1")
    mmSetParamValue("::scs::udptarget", "127.0.0.1")
    -- use this only if the clients are mpiclusterviews -> they do not need the broadcast
    mmSetParamValue("::scs::server::noEcho", "true")
else
    mmCreateModule("View3D", "::mini-lua::v")
end
mmCreateModule("SimpleSphereRenderer", "::mini-lua::ssr")
mmSetParamValue("::mini-lua::ssr::renderMode", "Simple")
mmCreateModule("MMPLDDataSource", "::mini-lua::ds")
mmCreateModule("LinearTransferFunction", "::mini-lua::ltf")
mmCreateCall("CallRender3D", "::mini-lua::v::rendering", "::mini-lua::ssr::rendering")
mmCreateCall("MultiParticleDataCall", "::mini-lua::ssr::getdata", "::mini-lua::ds::getdata")
mmCreateCall("CallGetTransferFunction", "::mini-lua::ssr::gettransferfunction", "::mini-lua::ltf::gettransferfunction")
mmSetParamValue("::mini-lua::ds::filename", "S:\\Projekte\\SFB 716\\D3\\Data\\riss.mmpld")
