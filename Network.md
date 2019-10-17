# Lua-based Projects and distributed / Powerwall-enable MegaMol

*TODO: more explanation.*

Every cluster node just needs a generic tile to display:

```
mmCreateView("t1", "MpiClusterView", "::t1::mcview")
mmCreateModule("SimpleClusterClient", "::mcc")
mmCreateModule("MpiProvider", "::mpi")
mmCreateCall("SimpleClusterClientViewRegistration", "::t1::mcview::register", "::mcc::registerView")
mmCreateCall("MpiCall", "::t1::mcview::requestMpi", "::mpi::provideMpi")
```

this file is then executed via
```
mmconsole.exe -p ..\..\megamol-prj\mpiclusterview.lua
```

The control machine needs a nearly normal project file like this:

```
role = mmGetConfigValue("role")
print("I am a " .. role)
if role == "boss" then
    mmCreateView("mini-lua", "View3D", "v")
    mmCreateJob("simpleclusterserver", "SimpleClusterServer", "::scs")
    mmSetParamValue("::scs::viewname", "::mini-lua::v")
    mmSetParamValue("::scs::server::Name", "127.0.0.1")
    mmSetParamValue("::scs::udptarget", "127.0.0.1")
else
    mmCreateModule("View3D", "::mini-lua::v")
end
mmCreateModule("SphererRenderer", "::mini-lua::ssr")
mmCreateModule("MMPLDDataSource", "::mini-lua::ds")
mmCreateModule("LinearTransferFunction", "::mini-lua::ltf")
mmCreateCall("CallRender3D", "::mini-lua::v::rendering", "::mini-lua::ssr::rendering")
mmCreateCall("MultiParticleDataCall", "::mini-lua::ssr::getdata", "::mini-lua::ds::getdata")
mmCreateCall("CallGetTransferFunction", "::mini-lua::ssr::gettransferfunction", "::mini-lua::ltf::gettransferfunction")
mmSetParamValue("::mini-lua::ds::filename", "S:\\Projekte\\SFB 716\\D3\\Data\\riss.mmpld")

```

which is executed like so:

```
mmconsole -p ..\..\megamol-prj\mini-lua.mmprj.lua -o role boss
```