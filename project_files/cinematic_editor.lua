
fileToRender = "../project_files/testspheres.lua"
keyframeFile = "../project_files/cinematic_keyframes.kf"



function trafo(str)
  local newcontent = str:gsub('mmCreateView%(.-%)', "")
  local viewname, viewclass, viewmoduleinst = str:match(
      'mmCreateView%(%s*[\"\']([^\"\']+)[\"\']%s*,%s*[\"\']([^\"\']+)[\"\']%s*,%s*[\"\']([^\"\']+)[\"\']%s*%)')

  print("viewname = " .. viewname)
  print("viewclass = " .. viewclass)
  print("viewmoduleinst = " .. viewmoduleinst)

  newcontent = newcontent:gsub('mmCreateCall%([\"\']CallRender3D[\'\"],%s*[\'\"]' 
      .. '.-' .. viewmoduleinst .. '::rendering[\'\"],([^,]+)%)', 'mmCreateCall("CallRender3D", "::project::ReplacementRenderer1::renderer",%1)'
      .. "\n" .. 'mmCreateCall("CallRender3D", "::project::ReplacementRenderer2::renderer",%1)')
  
  return newcontent
end

local content = mmReadTextFile(fileToRender, trafo)

print("read: " .. content)
code = load(content)
code()



mmCreateView("project", "SplitView", "SplitView1")
mmSetParamValue("::project::SplitView1::split.orientation", "1")
mmSetParamValue("::project::SplitView1::split.pos", "0.65")
mmSetParamValue("::project::SplitView1::split.colour", "gray")

mmCreateModule("SplitView", "::project::SplitView2")
mmSetParamValue("::project::SplitView2::split.pos", "0.55")
mmSetParamValue("::project::SplitView2::split.colour", "gray")

mmCreateModule("KeyframeKeeper", "::project::KeyframeKeeper1")
mmSetParamValue("::project::KeyframeKeeper1::storage::01_filename", keyframeFile)

mmCreateModule("View2D", "::project::View2D1")
mmSetParamValue("::project::View2D1::backCol", "black")
mmSetParamValue("::project::View2D1::resetViewOnBBoxChange", "True")

mmCreateModule("View3D", "::project::View3D1")
mmSetParamValue("::project::View3D1::backCol", "black")
mmSetParamValue("::project::View3D1::bboxCol", "gray")

mmCreateModule("TimeLineRenderer", "::project::TimeLineRenderer1")

mmCreateModule("CinematicRenderer", "::project::CinematicRenderer1")

mmCreateModule("CinematicView", "::project::CinematicView1")
mmSetParamValue("::project::CinematicView1::backCol", "grey")
mmSetParamValue("::project::CinematicView1::bboxCol", "white")
mmSetParamValue("::project::CinematicView1::06_fps", "24")
mmSetParamValue("::project::CinematicView1::stereo::projection", "2")

mmCreateModule("ReplacementRenderer", "::project::ReplacementRenderer1")
mmSetParamValue("::project::ReplacementRenderer1::03_replacmentKeyAssign", "6")
mmSetParamValue("::project::ReplacementRenderer1::01_replacementRendering", "on")

mmCreateModule("ReplacementRenderer", "::project::ReplacementRenderer2")
mmSetParamValue("::project::ReplacementRenderer2::03_replacmentKeyAssign", "5")
mmSetParamValue("::project::ReplacementRenderer2::01_replacementRendering", "on")


mmCreateCall("CallRenderView", "::project::SplitView1::render1", "::project::SplitView2::render")
mmCreateCall("CallRenderView", "::project::SplitView1::render2", "::project::View2D1::render")
mmCreateCall("CallRenderView", "::project::SplitView2::render1", "::project::View3D1::render")
mmCreateCall("CallCinematicCamera", "::project::TimeLineRenderer1::getkeyframes", "::project::KeyframeKeeper1::scene3D")
mmCreateCall("CallRender3D", "::project::View3D1::rendering", "::project::CinematicRenderer1::rendering")
mmCreateCall("CallCinematicCamera", "::project::CinematicRenderer1::keyframeKeeper", "::project::KeyframeKeeper1::scene3D")
mmCreateCall("CallRender3D", "::project::CinematicRenderer1::renderer", "::project::ReplacementRenderer1::rendering")
mmCreateCall("CallRender2D", "::project::View2D1::rendering", "::project::TimeLineRenderer1::rendering")
mmCreateCall("CallRenderView", "::project::SplitView2::render2", "::project::CinematicView1::render")
mmCreateCall("CallCinematicCamera", "::project::CinematicView1::keyframeKeeper", "::project::KeyframeKeeper1::scene3D")
mmCreateCall("CallRender3D", "::project::CinematicView1::rendering", "::project::ReplacementRenderer2::rendering")
