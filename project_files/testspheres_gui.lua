
mmCreateView("testspheres", "GUIView", "::guiview")
mmCreateModule("View3D", "::view")
mmCreateModule("SphereRenderer", "::rnd")
mmSetParamValue("rnd::renderMode", "Simple")
mmCreateModule("TestSpheresDataSource", "::dat")
mmCreateCall("CallRenderView", "::guiview::renderview", "::view::render")
mmCreateCall("CallRender3D", "::view::rendering", "::rnd::rendering")
mmCreateCall("MultiParticleDataCall", "::rnd::getData", "::dat::getData")
