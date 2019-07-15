
fileToRender = "../project_files/testspheres.lua"
keyframeFile = "../project_files/cinematic_keyframes.kf"


function trafo(str)

  -- Break if SplitView or CinematicView occure anywhere in the project
  local startpos, endpos, word = str:find("SplitView")
  if not (startpos == nil) then
    print( "lua ERROR: Cinematic Editor can not be used with projects containing \"SplitView\"."  )
    return ""
  end
  startpos, endpos, word = str:find("CinematicView")
  if not (startpos == nil) then
    print( "lua ERROR: Cinematic Editor can not be used with projects containing \"CinematicView\"."  )
    return ""
  end  

  local viewclass, viewmoduleinst
  startpos, endpos, word = str:find("mmCreateView%(.-,%s*[\"\']([^\"\']+)[\"\']%s*,.-%)")
  if word == "GUIView" then
    print( "lua INFO: Found \"GUIView\" as head view." )
    startpos, endpos, word = str:find("mmCreateModule%(.-View.-%)")
    local substring = str:sub(startpos, endpos)
    viewclass, viewmoduleinst = substring:match(
      'mmCreateModule%(%s*[\"\']([^\"\']+)[\"\']%s*,%s*[\"\']([^\"\']+)[\"\']%s*%)')
  else
     viewclass, viewmoduleinst = str:match(
      'mmCreateView%(.-,%s*[\"\']([^\"\']+)[\"\']%s*,%s*[\"\']([^\"\']+)[\"\']%s*%)')
  end
  print("lua INFO: View Class = " .. viewclass)
  print("lua INFO: View Module Instance = " .. viewmoduleinst)

  local newcontent  = str:gsub("mmCreateView%(.-%)", "")
  newcontent = newcontent:gsub("mmCreateModule%(.-\"View.-%)", "")
  newcontent = newcontent:gsub("mmCreateCall%(\"CallRenderView.-%)", "")
  newcontent = newcontent:gsub('mmCreateCall%([\"\']CallRender3D[\'\"],%s*[\'\"]' 
      .. '.-' .. viewmoduleinst .. '::rendering[\'\"],([^,]+)%)', 'mmCreateCall("CallRender3D", "::ReplacementRenderer1::renderer",%1)'
      .. "\n" .. 'mmCreateCall("CallRender3D", "::ReplacementRenderer2::renderer",%1)')
  
  return newcontent
end


local content = mmReadTextFile(fileToRender, trafo)
print("lua INFO: Transformed Include Project File =\n" .. content .. "\n\n")
code = load(content)
code()


mmCreateView("testeditor", "GUIView", "::GUIView1")
mmCreateModule("SplitView", "::SplitView1")
mmSetParamValue("::SplitView1::split.orientation", "1")
mmSetParamValue("::SplitView1::split.pos", "0.65")
mmSetParamValue("::SplitView1::split.colour", "gray")

mmCreateModule("SplitView", "::SplitView2")
mmSetParamValue("::SplitView2::split.pos", "0.55")
mmSetParamValue("::SplitView2::split.colour", "gray")

mmCreateModule("KeyframeKeeper", "::KeyframeKeeper1")
mmSetParamValue("::KeyframeKeeper1::storage::filename", keyframeFile)

mmCreateModule("View2D", "::View2D1")
mmSetParamValue("::View2D1::backCol", "black")
mmSetParamValue("::View2D1::resetViewOnBBoxChange", "True")

mmCreateModule("View3D", "::View3D1")
mmSetParamValue("::View3D1::backCol", "black")
mmSetParamValue("::View3D1::bboxCol", "gray")

mmCreateModule("TimeLineRenderer", "::TimeLineRenderer1")

mmCreateModule("TrackingshotRenderer", "::TrackingshotRenderer1")

mmCreateModule("CinematicView", "::CinematicView1")
mmSetParamValue("::CinematicView1::backCol", "grey")
mmSetParamValue("::CinematicView1::bboxCol", "white")
mmSetParamValue("::CinematicView1::fps", "24")
mmSetParamValue("::CinematicView1::stereo::projection", "2")

mmCreateModule("ReplacementRenderer", "::ReplacementRenderer1")
mmSetParamValue("::ReplacementRenderer1::replacmentKeyAssign", "4")
mmSetParamValue("::ReplacementRenderer1::replacementRendering", "on")

mmCreateModule("ReplacementRenderer", "::ReplacementRenderer2")
mmSetParamValue("::ReplacementRenderer2::replacmentKeyAssign", "3")
mmSetParamValue("::ReplacementRenderer2::replacementRendering", "off")


mmCreateCall("CallRenderView", "::GUIView1::renderview", "::SplitView1::render")
mmCreateCall("CallRenderView", "::SplitView1::render1", "::SplitView2::render")
mmCreateCall("CallRenderView", "::SplitView1::render2", "::View2D1::render")
mmCreateCall("CallRenderView", "::SplitView2::render1", "::View3D1::render")
mmCreateCall("CallKeyframeKeeper", "::TimeLineRenderer1::getkeyframes", "::KeyframeKeeper1::scene3D")
mmCreateCall("CallRender3D", "::View3D1::rendering", "::TrackingshotRenderer1::rendering")
mmCreateCall("CallKeyframeKeeper", "::TrackingshotRenderer1::keyframeKeeper", "::KeyframeKeeper1::scene3D")
mmCreateCall("CallRender3D", "::TrackingshotRenderer1::renderer", "::ReplacementRenderer1::rendering")
mmCreateCall("CallRender2D", "::View2D1::rendering", "::TimeLineRenderer1::rendering")
mmCreateCall("CallRenderView", "::SplitView2::render2", "::CinematicView1::render")
mmCreateCall("CallKeyframeKeeper", "::CinematicView1::keyframeKeeper", "::KeyframeKeeper1::scene3D")
mmCreateCall("CallRender3D", "::CinematicView1::rendering", "::ReplacementRenderer2::rendering")
