/*
 * View3D2000GT.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/View3D2000GT.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#ifdef _WIN32
#    include <windows.h>
#endif /* _WIN32 */
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/CallRender3D.h"   // TODO new call?
#include "mmcore/view/CallRenderView.h" // TODO new call?
#include "mmcore/view/CameraParamOverride.h"
#include "vislib/Exception.h"
#include "vislib/String.h"
#include "vislib/StringSerialiser.h"
#include "vislib/Trace.h"
#include "vislib/graphics/BitmapImage.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/math/Point.h"
#include "vislib/math/Quaternion.h"
#include "vislib/math/Vector.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/KeyCode.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/sysfunctions.h"

using namespace megamol::core;
using namespace megamol::core::view;

/*
 * View3D2000GT::View3D2000GT
 */
View3D2000GT::View3D2000GT(void)
    : view::AbstractView3D()
    , AbstractCamParamSync()
    , cam()
    , camParams()
    , camOverrides()
    , cursor2d()
    , modkeys()
    , rotator1()
    , rotator2()
    , zoomer1()
    , zoomer2()
    , mover()
    , lookAtDist()
    , rendererSlot("rendering", "Connects the view to a Renderer")
    , lightDir(0.5f, -1.0f, -1.0f)
    , isCamLight(true)
    , bboxs()
    , showBBox("showBBox", "Bool parameter to show/hide the bounding box")
    , showLookAt("showLookAt", "Flag showing the look at point")
    , cameraSettingsSlot("camsettings", "The stored camera settings")
    , storeCameraSettingsSlot("storecam", "Triggers the storage of the camera settings")
    , restoreCameraSettingsSlot("restorecam", "Triggers the restore of the camera settings")
    , resetViewSlot("resetView", "Triggers the reset of the view")
    , firstImg(false)
    , frozenValues(NULL)
    , isCamLightSlot(
          "light::isCamLight", "Flag whether the light is relative to the camera or to the world coordinate system")
    , lightDirSlot("light::direction", "Direction vector of the light")
    , lightColDifSlot("light::diffuseCol", "Diffuse light colour")
    , lightColAmbSlot("light::ambientCol", "Ambient light colour")
    , stereoFocusDistSlot("stereo::focusDist", "focus distance for stereo projection")
    , stereoEyeDistSlot("stereo::eyeDist", "eye distance for stereo projection")
    , overrideCall(NULL)
    , viewKeyMoveStepSlot("viewKey::MoveStep", "The move step size in world coordinates")
    , viewKeyRunFactorSlot("viewKey::RunFactor", "The factor for step size multiplication when running (shift)")
    , viewKeyAngleStepSlot("viewKey::AngleStep", "The angle rotate step in degrees")
    , mouseSensitivitySlot("viewKey::MouseSensitivity", "used for WASD mode")
    , viewKeyRotPointSlot("viewKey::RotPoint", "The point around which the view will be roateted")
    , viewKeyRotLeftSlot("viewKey::RotLeft", "Rotates the view to the left (around the up-axis)")
    , viewKeyRotRightSlot("viewKey::RotRight", "Rotates the view to the right (around the up-axis)")
    , viewKeyRotUpSlot("viewKey::RotUp", "Rotates the view to the top (around the right-axis)")
    , viewKeyRotDownSlot("viewKey::RotDown", "Rotates the view to the bottom (around the right-axis)")
    , viewKeyRollLeftSlot("viewKey::RollLeft", "Rotates the view counter-clockwise (around the view-axis)")
    , viewKeyRollRightSlot("viewKey::RollRight", "Rotates the view clockwise (around the view-axis)")
    , viewKeyZoomInSlot("viewKey::ZoomIn", "Zooms in (moves the camera)")
    , viewKeyZoomOutSlot("viewKey::ZoomOut", "Zooms out (moves the camera)")
    , viewKeyMoveLeftSlot("viewKey::MoveLeft", "Moves to the left")
    , viewKeyMoveRightSlot("viewKey::MoveRight", "Moves to the right")
    , viewKeyMoveUpSlot("viewKey::MoveUp", "Moves to the top")
    , viewKeyMoveDownSlot("viewKey::MoveDown", "Moves to the bottom")
    , toggleBBoxSlot("toggleBBox", "Button to toggle the bounding box")
    , toggleSoftCursorSlot("toggleSoftCursor", "Button to toggle the soft cursor")
    , bboxCol{1.0f, 1.0f, 1.0f, 0.625f}
    , bboxColSlot("bboxCol", "Sets the colour for the bounding box")
    , enableMouseSelectionSlot("enableMouseSelection", "Enable selecting and picking with the mouse")
    , showViewCubeSlot("viewcube::show", "Shows the view cube helper")
    , resetViewOnBBoxChangeSlot("resetViewOnBBoxChange", "whether to reset the view when the bounding boxes change")
    , mouseX(0.0f)
    , mouseY(0.0f)
    , mouseFlags(0)
    , timeCtrl()
    , toggleMouseSelection(false)
    , paramAlpha("watermark::01_alpha", "The alpha value for the watermarks.")
    , paramScaleAll("watermark::02_scaleAll", "The scale factor for all images.")
    , paramImgTopLeft("watermark::03_imageTopLeft", "The image file name for the top left watermark.")
    , paramScaleTopLeft("watermark::04_scaleTopLeft", "The scale factor for the top left watermark.")
    , paramImgTopRight("watermark::05_imageTopRight", "The image file name for the top right watermark.")
    , paramScaleTopRight("watermark::06_scaleTopRight", "The scale factor for the top right watermark.")
    , paramImgBottomLeft("watermark::07_imageBottomLeft", "The image file name for the bottom left watermark.")
    , paramScaleBottomLeft("watermark::08_scaleBottomLeft", "The scale factor for the botttom left watermark.")
    , paramImgBottomRight("watermark::09_imageBottomRight", "The image file name for the bottom right watermark.")
    , paramScaleBottomRight("watermark::10_scaleBottomRight", "The scale factor for the bottom right watermark.")
    , hookOnChangeOnlySlot("hookOnChange", "whether post-hooks are triggered when the frame would be identical")
    , textureBottomLeft()
    , textureBottomRight()
    , textureTopLeft()
    , textureTopRight() {

    using vislib::sys::KeyCode;

    // TODO implement
}

/*
 * View3D2000GT::~View3D2000GT
 */
View3D2000GT::~View3D2000GT(void) {
    this->Release();
    SAFE_DELETE(this->frozenValues);
    this->overrideCall = nullptr; // DO NOT DELETE
}

/*
 * View3D2000GT::GetCameraSyncNumber
 */
unsigned int view::View3D2000GT::GetCameraSyncNumber(void) const {
    // TODO implement
    return 0;
}


/*
 * View3D2000GT::SerialiseCamera
 */
void view::View3D2000GT::SerialiseCamera(vislib::Serialiser& serialiser) const {
    // TODO implement
    this->camParams->Serialise(serialiser);
}


/*
 * View3D2000GT::DeserialiseCamera
 */
void view::View3D2000GT::DeserialiseCamera(vislib::Serialiser& serialiser) {
    // TODO implement
    this->camParams->Deserialise(serialiser);
}

/*
 * View3D2000GT::Render
 */
void View3D2000GT::Render(const mmcRenderViewContext& context) {
    // TODO implement
}

/*
 * View3D2000GT::ResetView
 */
void View3D2000GT::ResetView(void) {
    // TODO implement
}

/*
 * View3D2000GT::Resize
 */
void View3D2000GT::Resize(unsigned int width, unsigned int height) {
    // TODO implement
}

/*
 * View3D2000GT::SetCursor2DButtonState
 */
void View3D2000GT::SetCursor2DButtonState(unsigned int btn, bool down) {
    // TODO implement
}

/*
 * View3D2000GT::SetCursor2DPosition
 */
void View3D2000GT::SetCursor2DPosition(float x, float y) {
    // TODO implement
}

/*
 * View3D2000GT::SetInputModifier
 */
void View3D2000GT::SetInputModifier(mmcInputModifier mod, bool down) {
    // TODO implement
}

/*
 * View3D2000GT::OnRenderView
 */
bool View3D2000GT::OnRenderView(Call& call) {
    // TODO implement
    return true;
}

/*
 * View3D2000GT::UpdateFreeze
 */
void View3D2000GT::UpdateFreeze(bool freeze) {
    // TODO implement
}

/*
 * View3D2000GT::unpackMouseCoordinates
 */
void View3D2000GT::unpackMouseCoordinates(float& x, float& y) {
    // TODO implement
}

/*
 * View3D2000GT::create
 */
bool View3D2000GT::create(void) {
    // TODO implement
    return true;
}

/*
 * View3D2000GT::release
 */
void View3D2000GT::release(void) {
    this->removeTitleRenderer();
    this->cursor2d.UnregisterCursorEvent(&this->rotator1);
    this->cursor2d.UnregisterCursorEvent(&this->rotator2);
    this->cursor2d.UnregisterCursorEvent(&this->zoomer1);
    this->cursor2d.UnregisterCursorEvent(&this->zoomer2);
    this->cursor2d.UnregisterCursorEvent(&this->mover);
    SAFE_DELETE(this->frozenValues);
}

/*
 * View3D2000GT::mouseSensitivityChanged
 */
bool View3D2000GT::mouseSensitivityChanged(param::ParamSlot& p) {
    this->rotator2.SetMouseSensitivity(p.Param<param::FloatParam>()->Value());
    return true;
}

/*
 * View3D2000GT::renderBBox
 */
void View3D2000GT::renderBBox(void) {
    // TODO implement
}

/*
 * View3D2000GT::renderBBoxBackside
 */
void View3D2000GT::renderBBoxBackside(void) {
    // TODO implement
}

/*
 * View3D2000GT::renderBBoxFrontside
 */
void View3D2000GT::renderBBoxFrontside(void) {
    // TODO implement
}

/*
 * View3D2000GT::renderLookAt
 */
void View3D2000GT::renderLookAt(void) {
    // TODO implement
}

/*
 * View3D2000GT::renderSoftCursor
 */
void View3D2000GT::renderSoftCursor(void) {
    // TODO implement
}

/*
 * View3D2000GT::OnGetCamParams
 */
bool View3D2000GT::OnGetCamParams(CallCamParamSync& c) {
    // TODO implement
    return true;
}

/*
 * View3D2000GT::onStoreCamera
 */
bool View3D2000GT::onStoreCamera(param::ParamSlot& p) {
    // TODO implement
    return true;
}

/*
 * View3D2000GT::onRestoreCamera
 */
bool View3D2000GT::onRestoreCamera(param::ParamSlot& p) {
    // TODO implement
    return true;
}

/*
 * View3D2000GT::onResetView
 */
bool View3D2000GT::onResetView(param::ParamSlot& p) {
    this->ResetView();
    return true;
}

/*
 * View3D2000GT::viewKeyPressed
 */
bool View3D2000GT::viewKeyPressed(param::ParamSlot& p) {
    // TODO implement
    return true;
}

/*
 * View3D2000GT::onToggleButton
 */
bool View3D2000GT::onToggleButton(param::ParamSlot& p) {
    // TODO implement
    return true;
}

/*
 * View3D2000GT::renderViewCube
 */
void View3D2000GT::renderViewCube(void) {
    // TODO implement
}

/*
 * View3D2000GT::renderWatermark
 */
bool View3D2000GT::renderWatermark(View3D2000GT::corner, float vpH, float vpW) {
    // TODO implement
    return true;
}

/*
 * View3D2000GT::loadTexture
 */
bool View3D2000GT::loadTexture(View3D2000GT::corner, vislib::StringA filename) {
    // TODO implement
    return true;
}

/*
 * View3D2000GT::loadFile
 */
SIZE_T View3D2000GT::loadFile(vislib::StringA name, void** outData) {
    // TODO implement
    return 0;
}
