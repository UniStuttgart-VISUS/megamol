/*
 * AbstractView.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmstd/view/AbstractView.h"

#include <climits>
#include <fstream>

#include "mmcore/AbstractNamedObject.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/LongestEdgeCameraSamples.h"
#include "mmcore/utility/OrbitalCameraSamples.h"
#include "mmstd/renderer/AbstractCallRender.h"
#include "mmstd/renderer/CallRenderView.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/assert.h"

using namespace megamol::core;
using megamol::core::utility::log::Log;


/*
 * view::AbstractView::AbstractView
 */
view::AbstractView::AbstractView(ViewDimension dim)
        : AbstractViewInterface(dim)
        , _firstImg(false)
        , _cameraIsMutable(true)
        , _rhsRenderSlot("rendering", "Connects the view to a Renderer")
        , _lhsRenderSlot("render", "Connects modules requesting renderings")
        , _cameraSettingsSlot("camstore::settings", "Holds the camera settings of the currently stored camera.")
        , _storeCameraSettingsSlot("camstore::storecam",
              "Triggers the storage of the camera settings. This only works for "
              "multiple cameras if you use .lua project files")
        , _restoreCameraSettingsSlot("camstore::restorecam",
              "Triggers the restore of the camera settings. This only works "
              "for multiple cameras if you use .lua project files")
        , _overrideCamSettingsSlot("camstore::overrideSettings",
              "When activated, existing camera settings files will be overwritten by this "
              "module. This only works if you use .lua project files")
        , _autoSaveCamSettingsSlot("camstore::autoSaveSettings",
              "When activated, the camera settings will be stored to disk whenever a camera checkpoint is saved or "
              "MegaMol "
              "is closed. This only works if you use .lua project files")
        , _autoLoadCamSettingsSlot("camstore::autoLoadSettings",
              "When activated, the view will load the camera settings from disk at startup. "
              "This only works if you use .lua project files")
        , _resetViewSlot("view::resetView", "Triggers the reset of the view")
        , _resetViewOnBBoxChangeSlot(
              "resetViewOnBBoxChange", "whether to reset the view when the bounding boxes change")
        , _showLookAt("showLookAt", "Flag showing the look at point")
        , _showViewCubeParam("view::showViewCube", "Shows view cube.")
        , _hooks()
        , _timeCtrl()
        , _backgroundColSlot("backCol", "The views background colour") {
    // InputCall
    this->_lhsRenderSlot.SetCallback(
        view::CallRenderView::ClassName(), InputCall::FunctionName(InputCall::FnOnKey), &AbstractView::OnKeyCallback);
    this->_lhsRenderSlot.SetCallback(
        view::CallRenderView::ClassName(), InputCall::FunctionName(InputCall::FnOnChar), &AbstractView::OnCharCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(),
        InputCall::FunctionName(InputCall::FnOnMouseButton), &AbstractView::OnMouseButtonCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(),
        InputCall::FunctionName(InputCall::FnOnMouseMove), &AbstractView::OnMouseMoveCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(),
        InputCall::FunctionName(InputCall::FnOnMouseScroll), &AbstractView::OnMouseScrollCallback);
    // AbstractCallRender
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnRender), &AbstractView::OnRenderView);
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnGetExtents), &AbstractView::GetExtents);
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(),
        view::CallRenderView::FunctionName(view::CallRenderView::CALL_RESETVIEW), &AbstractView::OnResetView);
    // this->MakeSlotAvailable(&this->renderSlot);

    this->_cameraSettingsSlot.SetParameter(new param::StringParam(""));
    this->MakeSlotAvailable(&this->_cameraSettingsSlot);

    this->_storeCameraSettingsSlot.SetParameter(
        new param::ButtonParam(view::Key::KEY_C, (view::Modifier::SHIFT | view::Modifier::ALT)));
    this->_storeCameraSettingsSlot.SetUpdateCallback(&AbstractView::onStoreCamera);
    this->MakeSlotAvailable(&this->_storeCameraSettingsSlot);

    this->_restoreCameraSettingsSlot.SetParameter(new param::ButtonParam(view::Key::KEY_C, view::Modifier::ALT));
    this->_restoreCameraSettingsSlot.SetUpdateCallback(&AbstractView::onRestoreCamera);
    this->MakeSlotAvailable(&this->_restoreCameraSettingsSlot);

    this->_overrideCamSettingsSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->_overrideCamSettingsSlot);

    this->_autoSaveCamSettingsSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->_autoSaveCamSettingsSlot);

    this->_autoLoadCamSettingsSlot.SetParameter(new param::BoolParam(true));
    this->MakeSlotAvailable(&this->_autoLoadCamSettingsSlot);

    this->_resetViewSlot.SetParameter(new param::ButtonParam(Key::KEY_HOME));
    this->_resetViewSlot.SetUpdateCallback(&AbstractView::OnResetView);
    this->MakeSlotAvailable(&this->_resetViewSlot);

    this->_resetViewOnBBoxChangeSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->_resetViewOnBBoxChangeSlot);

    this->_showLookAt.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->_showLookAt);

    this->_showViewCubeParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->_showViewCubeParam);

    for (unsigned int i = 0; this->_timeCtrl.GetSlot(i) != NULL; i++) {
        this->MakeSlotAvailable(this->_timeCtrl.GetSlot(i));
    }

    // this triggers the initialization
    this->_bboxs.Clear();

    this->_backgroundCol[0] = 0.0f;
    this->_backgroundCol[1] = 0.0f;
    this->_backgroundCol[2] = 0.125f;
    this->_backgroundCol[3] = 1.0f;

    this->_backgroundColSlot << new param::ColorParam(
        this->_backgroundCol[0], this->_backgroundCol[1], this->_backgroundCol[2], this->_backgroundCol[3]);
    this->MakeSlotAvailable(&this->_backgroundColSlot);
}


/*
 * view::AbstractView::~AbstractView
 */
view::AbstractView::~AbstractView(void) {
    this->_hooks.Clear(); // DO NOT DELETE OBJECTS
}


/*
 * view::AbstractView::IsParamRelevant
 */
bool view::AbstractView::IsParamRelevant(const std::shared_ptr<param::AbstractParam>& param) const {
    const AbstractNamedObject* ano = dynamic_cast<const AbstractNamedObject*>(this);
    if (ano == NULL)
        return false;
    if (param == nullptr)
        return false;

    vislib::SingleLinkedList<const AbstractNamedObject*> searched;
    return ano->IsParamRelevant(searched, param);
}

void megamol::core::view::AbstractView::SetCamera(Camera camera, bool isMutable) {
    _camera = camera;
    _cameraIsMutable = isMutable;
}

megamol::core::view::Camera megamol::core::view::AbstractView::GetCamera() const {
    return _camera;
}

void megamol::core::view::AbstractView::CalcCameraClippingPlanes(float border) {
    if (_cameraIsMutable) {
        auto cam_pose = _camera.get<Camera::Pose>();
        glm::vec3 front = cam_pose.direction;
        glm::vec3 pos = cam_pose.position;

        float dist, minDist, maxDist;

        dist = glm::dot(front, glm::make_vec3(_bboxs.ClipBox().GetLeftBottomBack().PeekCoordinates()) - pos);
        minDist = maxDist = dist;

        dist = glm::dot(front, glm::make_vec3(_bboxs.ClipBox().GetLeftBottomFront().PeekCoordinates()) - pos);
        if (dist < minDist)
            minDist = dist;
        if (dist > maxDist)
            maxDist = dist;

        dist = glm::dot(front, glm::make_vec3(_bboxs.ClipBox().GetLeftTopBack().PeekCoordinates()) - pos);
        if (dist < minDist)
            minDist = dist;
        if (dist > maxDist)
            maxDist = dist;

        dist = glm::dot(front, glm::make_vec3(_bboxs.ClipBox().GetLeftTopFront().PeekCoordinates()) - pos);
        if (dist < minDist)
            minDist = dist;
        if (dist > maxDist)
            maxDist = dist;

        dist = glm::dot(front, glm::make_vec3(_bboxs.ClipBox().GetRightBottomBack().PeekCoordinates()) - pos);
        if (dist < minDist)
            minDist = dist;
        if (dist > maxDist)
            maxDist = dist;

        dist = glm::dot(front, glm::make_vec3(_bboxs.ClipBox().GetRightBottomFront().PeekCoordinates()) - pos);
        if (dist < minDist)
            minDist = dist;
        if (dist > maxDist)
            maxDist = dist;

        dist = glm::dot(front, glm::make_vec3(_bboxs.ClipBox().GetRightTopBack().PeekCoordinates()) - pos);
        if (dist < minDist)
            minDist = dist;
        if (dist > maxDist)
            maxDist = dist;

        dist = glm::dot(front, glm::make_vec3(_bboxs.ClipBox().GetRightTopFront().PeekCoordinates()) - pos);
        if (dist < minDist)
            minDist = dist;
        if (dist > maxDist)
            maxDist = dist;

        minDist -= border;
        maxDist += border;

        // since the minDist is broken, we fix it here
        minDist = maxDist * 0.001f;

        auto cam_proj_type = _camera.get<Camera::ProjectionType>();

        if (cam_proj_type == Camera::ProjectionType::PERSPECTIVE) {
            auto cam_intrinsics = _camera.get<Camera::PerspectiveParameters>();
            if (!(std::abs(cam_intrinsics.near_plane - minDist) < 0.00001f) ||
                !(std::abs(cam_intrinsics.far_plane - maxDist) < 0.00001f)) {
                // TODO set intrinsics with minDist and maxDist
            }
        } else if (cam_proj_type == Camera::ProjectionType::ORTHOGRAPHIC) {
            auto cam_intrinsics = _camera.get<Camera::OrthographicParameters>();
            if (!(std::abs(cam_intrinsics.near_plane - minDist) < 0.00001f) ||
                !(std::abs(cam_intrinsics.far_plane - maxDist) < 0.00001f)) {
                // TODO set intrinsics with minDist and maxDist
            }
        } else {
            // print warning
        }
    }
}

/*
 * view::AbstractView::OnRenderView
 */
bool view::AbstractView::OnRenderView(Call& call) {
    throw vislib::UnsupportedOperationException("AbstractView::OnRenderView", __FILE__, __LINE__);
}

void megamol::core::view::AbstractView::beforeRender(double time, double instanceTime) {
    float simulationTime = static_cast<float>(time);

    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }

    AbstractCallRender* cr = this->_rhsRenderSlot.CallAs<AbstractCallRender>();

    auto bgCol = this->BackgroundColor();

    if (cr == NULL) {
        return; // empty enough
    }

    cr->SetBackgroundColor(glm::vec4(bgCol[0], bgCol[1], bgCol[2], bgCol[3]));

    if ((*cr)(AbstractCallRender::FnGetExtents)) {
        if (!(cr->AccessBoundingBoxes() == this->_bboxs) && cr->AccessBoundingBoxes().IsAnyValid()) {
            this->_bboxs = cr->AccessBoundingBoxes();
            glm::vec3 bbcenter = glm::make_vec3(this->_bboxs.BoundingBox().CalcCenter().PeekCoordinates());

            if (_resetViewOnBBoxChangeSlot.Param<param::BoolParam>()->Value()) {
                this->ResetView();
            }
        }

        if (this->_firstImg) {
            this->ResetView();
            this->_firstImg = false;
            if (this->_autoLoadCamSettingsSlot.Param<param::BoolParam>()->Value()) {
                this->onRestoreCamera(this->_restoreCameraSettingsSlot);
            }
            this->_lastFrameTime = std::chrono::high_resolution_clock::now();
        }

        this->_timeCtrl.SetTimeExtend(cr->TimeFramesCount(), false);
        if (simulationTime > static_cast<float>(cr->TimeFramesCount())) {
            simulationTime = static_cast<float>(cr->TimeFramesCount());
        }

        // old code was ...SetTime(this->frozenValues ? this->frozenValues->time : time);
        cr->SetTime(simulationTime);
    }

    // TODO
    // cr3d->SetCameraParameters(this->cam.Parameters()); // < here we use the 'active' parameters!
    // TODO!? cr3d->SetLastFrameTime(AbstractRenderingView::lastFrameTime());

    auto currentTime = std::chrono::high_resolution_clock::now();
    this->_lastFrameDuration =
        std::chrono::duration_cast<std::chrono::microseconds>(currentTime - this->_lastFrameTime);
    this->_lastFrameTime = currentTime;

    cr->SetLastFrameTime(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::time_point_cast<std::chrono::milliseconds>(this->_lastFrameTime).time_since_epoch())
                             .count());

    CalcCameraClippingPlanes(0.1f);
}

void megamol::core::view::AbstractView::afterRender() {
    // this->lastFrameParams->CopyFrom(this->OnGetCamParams, false);

    if (this->doHookCode()) {
        this->doAfterRenderHook();
    }
}

/*
 * view::AbstractView::OnResetView
 */
bool view::AbstractView::OnResetView(Call& call) {
    this->ResetView();
    return true;
}

/*
 * view::AbstractView::onResetView
 */
bool view::AbstractView::OnResetView(param::ParamSlot& p) {
    this->ResetView();
    return true;
}


bool view::AbstractView::GetExtents(Call& call) {
    throw vislib::UnsupportedOperationException("AbstractView::GetExtents", __FILE__, __LINE__);
    return false;
}

bool view::AbstractView::OnKeyCallback(Call& call) {
    try {
        view::AbstractCallRender& cr = dynamic_cast<view::AbstractCallRender&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == InputEvent::Tag::Key && "Callback invocation mismatched input event");
        return this->OnKey(evt.keyData.key, evt.keyData.action, evt.keyData.mods);
    } catch (...) {
        ASSERT("OnKeyCallback call cast failed\n");
    }
    return false;
}

bool view::AbstractView::OnCharCallback(Call& call) {
    try {
        view::AbstractCallRender& cr = dynamic_cast<view::AbstractCallRender&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == InputEvent::Tag::Char && "Callback invocation mismatched input event");
        return this->OnChar(evt.charData.codePoint);
    } catch (...) {
        ASSERT("OnCharCallback call cast failed\n");
    }
    return false;
}

bool view::AbstractView::OnMouseButtonCallback(Call& call) {
    try {
        view::AbstractCallRender& cr = dynamic_cast<view::AbstractCallRender&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == InputEvent::Tag::MouseButton && "Callback invocation mismatched input event");
        return this->OnMouseButton(evt.mouseButtonData.button, evt.mouseButtonData.action, evt.mouseButtonData.mods);
    } catch (...) {
        ASSERT("OnMouseButtonCallback call cast failed\n");
    }
    return false;
}

bool view::AbstractView::OnMouseMoveCallback(Call& call) {
    try {
        view::AbstractCallRender& cr = dynamic_cast<view::AbstractCallRender&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == InputEvent::Tag::MouseMove && "Callback invocation mismatched input event");
        return this->OnMouseMove(evt.mouseMoveData.x, evt.mouseMoveData.y);
    } catch (...) {
        ASSERT("OnMouseMoveCallback call cast failed\n");
    }
    return false;
}

bool view::AbstractView::OnMouseScrollCallback(Call& call) {
    try {
        view::AbstractCallRender& cr = dynamic_cast<view::AbstractCallRender&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == InputEvent::Tag::MouseScroll && "Callback invocation mismatched input event");
        return this->OnMouseScroll(evt.mouseScrollData.dx, evt.mouseScrollData.dy);
    } catch (...) {
        ASSERT("OnMouseScrollCallback call cast failed\n");
    }
    return false;
}

/*
 * AbstractView::onStoreCamera
 */
bool view::AbstractView::onStoreCamera(param::ParamSlot& p) {
    // save the current camera, too
    this->_savedCameras[10].first = _camera;
    this->_savedCameras[10].second = true;
    this->_cameraSerializer.setPrettyMode(false);
    std::string camstring = this->_cameraSerializer.serialize(this->_savedCameras[10].first);
    this->_cameraSettingsSlot.Param<param::StringParam>()->SetValue(camstring.c_str());

    auto path = this->determineCameraFilePath();
    if (path.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The camera output file path could not be determined. This is probably due to the usage of .mmprj project "
            "files. Please use a .lua project file instead");
        return false;
    }

    if (!this->_overrideCamSettingsSlot.Param<param::BoolParam>()->Value()) {
        // check if the file already exists
        std::ifstream file(path);
        if (file.good()) {
            file.close();
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "The camera output file path already contains a camera file with the name '%s'. Override mode is "
                "deactivated, so no camera is stored",
                path.c_str());
            return false;
        }
    }


    this->_cameraSerializer.setPrettyMode();
    auto outString = this->_cameraSerializer.serialize(this->_savedCameras);

    std::ofstream file(path);
    if (file.is_open()) {
        file << outString;
        file.close();
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The camera output file could not be written to '%s' because the file could not be opened.", path.c_str());
        return false;
    }

    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "Camera statistics successfully written to '%s'", path.c_str());
    return true;
}

/*
 * AbstractView::onRestoreCamera
 */
bool view::AbstractView::onRestoreCamera(param::ParamSlot& p) {
    if (!this->_cameraSettingsSlot.Param<param::StringParam>()->Value().empty()) {
        std::string camstring(this->_cameraSettingsSlot.Param<param::StringParam>()->Value());
        Camera cam;
        if (!this->_cameraSerializer.deserialize(cam, camstring)) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "The entered camera string was not valid. No change of the camera has been performed");
        } else {
            this->_camera = cam;
            return true;
        }
    }

    auto path = this->determineCameraFilePath();
    if (path.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The camera file path could not be determined. This is probably due to the usage of .mmprj project "
            "files. Please use a .lua project file instead");
        return false;
    }

    std::ifstream file(path);
    std::string text;
    if (file.is_open()) {
        text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The camera output file at '%s' could not be opened.", path.c_str());
        return false;
    }
    auto copy = this->_savedCameras;
    bool success = this->_cameraSerializer.deserialize(copy, text);
    if (!success) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The reading of the camera parameters did not work properly. No changes were made.");
        return false;
    }
    this->_savedCameras = copy;
    if (this->_savedCameras.back().second) {
        this->_camera = this->_savedCameras.back().first;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "The stored default cam was not valid. The old default cam is used");
    }
    return true;
}

/*
 * AbstractView::determineCameraFilePath
 */
std::string view::AbstractView::determineCameraFilePath(void) const {
    std::string path;

    const auto& paths = frontend_resources.get<megamol::frontend_resources::ScriptPaths>().lua_script_paths;
    if (!paths.empty()) {
        path = paths[0];
    } else {
        return path;
    }

    const auto dotpos = path.find_last_of('.');
    path = path.substr(0, dotpos);
    path.append("_cam.json");
    return path;
}

/*
 * view::AbstractView::BackgroundColor
 */
glm::vec4 view::AbstractView::BackgroundColor() const {
    if (this->_backgroundColSlot.IsDirty()) {
        this->_backgroundColSlot.ResetDirty();
        this->_backgroundColSlot.Param<param::ColorParam>()->Value(
            this->_backgroundCol.r, this->_backgroundCol.g, this->_backgroundCol.b, this->_backgroundCol.a);
    }
    return this->_backgroundCol;
}
