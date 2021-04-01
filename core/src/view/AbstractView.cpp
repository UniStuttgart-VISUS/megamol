/*
 * AbstractView.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/AbstractView.h"
#include <climits>
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/AbstractCallRender.h"
#include "vislib/assert.h"
#include "vislib/UnsupportedOperationException.h"

using namespace megamol::core;
using megamol::core::utility::log::Log;


/*
 * view::AbstractView::AbstractView
 */
view::AbstractView::AbstractView(void)
        : Module()
        , _firstImg(false)
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
        , _resetViewOnBBoxChangeSlot("resetViewOnBBoxChange", "whether to reset the view when the bounding boxes change")
        , _hooks()
        , _timeCtrl()
        , _bkgndColSlot("backCol", "The views background colour") {
    // InputCall
    this->_lhsRenderSlot.SetCallback(
        view::CallRenderView::ClassName(), InputCall::FunctionName(InputCall::FnOnKey), &AbstractView::OnKeyCallback);
    this->_lhsRenderSlot.SetCallback(
        view::CallRenderView::ClassName(), InputCall::FunctionName(InputCall::FnOnChar), &AbstractView::OnCharCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseButton),
        &AbstractView::OnMouseButtonCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseMove),
        &AbstractView::OnMouseMoveCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseScroll),
        &AbstractView::OnMouseScrollCallback);
    // AbstractCallRender
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnRender), &AbstractView::OnRenderView);
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnGetExtents), &AbstractView::GetExtents);
    // CallRenderView
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(),
        view::CallRenderView::FunctionName(view::CallRenderView::CALL_FREEZE), &AbstractView::OnFreezeView);
    this->_lhsRenderSlot.SetCallback(view::CallRenderView::ClassName(),
        view::CallRenderView::FunctionName(view::CallRenderView::CALL_UNFREEZE), &AbstractView::OnUnfreezeView);
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

    for (unsigned int i = 0; this->_timeCtrl.GetSlot(i) != NULL; i++) {
        this->MakeSlotAvailable(this->_timeCtrl.GetSlot(i));
    }

    // this triggers the initialization
    this->_bboxs.Clear();

    this->_bkgndCol[0] = 0.0f;
    this->_bkgndCol[1] = 0.0f;
    this->_bkgndCol[2] = 0.125f;
    this->_bkgndCol[3] = 0.0f;

    this->_bkgndColSlot << new param::ColorParam(this->_bkgndCol[0], this->_bkgndCol[1], this->_bkgndCol[2], 1.0f);
    this->MakeSlotAvailable(&this->_bkgndColSlot);
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
bool view::AbstractView::IsParamRelevant(
        const vislib::SmartPtr<param::AbstractParam>& param) const {
    const AbstractNamedObject* ano = dynamic_cast<const AbstractNamedObject*>(this);
    if (ano == NULL) return false;
    if (param.IsNull()) return false;

    vislib::SingleLinkedList<const AbstractNamedObject*> searched;
    return ano->IsParamRelevant(searched, param);
}


/*
 * view::AbstractView::DesiredWindowPosition
 */
bool view::AbstractView::DesiredWindowPosition(int *x, int *y, int *w,
        int *h, bool *nd) {
    Module *tm = dynamic_cast<Module*>(this);
    if (tm != NULL) {

        // this is not working properly if the main module/view is placed at top namespace root
        //vislib::StringA name(tm->Name());
        //if (tm->Parent() != NULL) name = tm->Parent()->Name();
        vislib::StringA name(tm->GetDemiRootName());

        if (name.IsEmpty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 1200,
                "View does not seem to have a name. Odd.");
        } else {
            name.Append("-Window");

            if (tm->GetCoreInstance()->Configuration().IsConfigValueSet(name)) {
                if (this->desiredWindowPosition(
                        tm->GetCoreInstance()->Configuration().ConfigValue(name),
                        x, y, w, h, nd)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 200,
                        "Loaded desired window geometry from \"%s\"", name.PeekBuffer());
                    return true;
                } else {
                    megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 200,
                        "Unable to load desired window geometry from \"%s\"", name.PeekBuffer());
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 1200,
                    "Unable to find window geometry settings \"%s\"", name.PeekBuffer());
            }
        }

        name = "*-Window";

        if (tm->GetCoreInstance()->Configuration().IsConfigValueSet(name)) {
            if (this->desiredWindowPosition(
                    tm->GetCoreInstance()->Configuration().ConfigValue(name),
                    x, y, w, h, nd)) {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 200,
                    "Loaded desired window geometry from \"%s\"", name.PeekBuffer());
                return true;
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 200,
                    "Unable to load desired window geometry from \"%s\"", name.PeekBuffer());
            }
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 1200,
                "Unable to find window geometry settings \"%s\"", name.PeekBuffer());
        }
    }

    return false;
}


/*
 * view::AbstractView::OnRenderView
 */
bool view::AbstractView::OnRenderView(Call& call) {
    throw vislib::UnsupportedOperationException(
        "AbstractView::OnRenderView", __FILE__, __LINE__);
}

/*
 * view::AbstractView::desiredWindowPosition
 */
bool view::AbstractView::desiredWindowPosition(const vislib::StringW& str,
        int *x, int *y, int *w, int *h, bool *nd) {
    vislib::StringW v = str;
    int vi = -1;
    v.TrimSpaces();

    if (x != NULL) { *x = INT_MIN; }
    if (y != NULL) { *y = INT_MIN; }
    if (w != NULL) { *w = INT_MIN; }
    if (h != NULL) { *h = INT_MIN; }
    if (nd != NULL) { *nd = false; }

    while (!v.IsEmpty()) {
        if ((v[0] == L'X') || (v[0] == L'x')) {
            vi = 0;
        } else if ((v[0] == L'Y') || (v[0] == L'y')) {
            vi = 1;
        } else if ((v[0] == L'W') || (v[0] == L'w')) {
            vi = 2;
        } else if ((v[0] == L'H') || (v[0] == L'h')) {
            vi = 3;
        } else if ((v[0] == L'N') || (v[0] == L'n')) {
            vi = 4;
        } else if ((v[0] == L'D') || (v[0] == L'd')) {
            if (nd != NULL) {
                *nd = (vi == 4);
            }
            vi = 4;
        } else {
            Log::DefaultLog.WriteMsg(
                megamol::core::utility::log::Log::LEVEL_WARN,
                "Unexpected character %s in window position definition.\n",
                vislib::StringA(vislib::StringA(v)[0], 1).PeekBuffer());
            break;
        }
        v = v.Substring(1);
        v.TrimSpaces();

        if (vi == 4) continue; // [n]d are not followed by a number

        if (vi >= 0) {
            // now we want to parse a double :-/
            int cp = 0;
            int len = v.Length();
            while ((cp < len) && (((v[cp] >= L'0') && (v[cp] <= L'9'))
                    || (v[cp] == L'+') /*|| (v[cp] == L'.')
                    || (v[cp] == L',') */|| (v[cp] == L'-')
                    /*|| (v[cp] == L'e') || (v[cp] == L'E')*/)) {
                cp++;
            }

            try {
                int i = vislib::CharTraitsW::ParseInt(v.Substring(0, cp));
                switch (vi) {
                    case 0 :
                        if (x != NULL) { *x = i; }
                        break;
                    case 1 :
                        if (y != NULL) { *y = i; }
                        break;
                    case 2 :
                        if (w != NULL) { *w = i; }
                        break;
                    case 3 :
                        if (h != NULL) { *h = i; }
                        break;
                }
            } catch(...) {
                const char *str = "unknown";
                switch (vi) {
                    case 0 : str = "X"; break;
                    case 1 : str = "Y"; break;
                    case 2 : str = "W"; break;
                    case 3 : str = "H"; break;
                }
                vi = -1;
                Log::DefaultLog.WriteMsg(
                    megamol::core::utility::log::Log::LEVEL_WARN,
                    "Unable to parse value for %s.\n", str);
            }

            v = v.Substring(cp);
        }

    }

    return true;
}

/*
 * AbstractView::Resize
 */
void view::AbstractView::Resize(unsigned int width, unsigned int height) {
    if (this->_camera.resolution_gate().width() != width || this->_camera.resolution_gate().height() != height) {
        this->_camera.resolution_gate(cam_type::screen_size_type(static_cast<LONG>(width), static_cast<LONG>(height)));
    }
    if (this->_camera.image_tile().width() != width || this->_camera.image_tile().height() != height) {
        this->_camera.image_tile(cam_type::screen_rectangle_type(
            std::array<int, 4>({0, static_cast<int>(height), static_cast<int>(width), 0})));
    }
}

void megamol::core::view::AbstractView::beforeRender(const mmcRenderViewContext& context) {
    float simulationTime = static_cast<float>(context.Time);
    float instTime = static_cast<float>(context.InstanceTime);

    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }

    glm::ivec4 currentViewport;
    AbstractCallRender* cr = this->_rhsRenderSlot.CallAs<AbstractCallRender>();

    auto bkgndCol = this->BkgndColour();

    if (cr == NULL) {
        return; // empty enough
    }

    cr->SetBackgroundColor(glm::vec4(bkgndCol[0], bkgndCol[1], bkgndCol[2], 0.0f));

    
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
    this->_lastFrameDuration = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - this->_lastFrameTime);
    this->_lastFrameTime = currentTime;

    cr->SetLastFrameTime(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::time_point_cast<std::chrono::milliseconds>(this->_lastFrameTime).time_since_epoch())
                               .count());

    this->_camera.CalcClipping(this->_bboxs.ClipBox(), 0.1f);
}

void megamol::core::view::AbstractView::afterRender(const mmcRenderViewContext& context) {
    // this->lastFrameParams->CopyFrom(this->OnGetCamParams, false);

    if (this->doHookCode()) {
        this->doAfterRenderHook();
    }
}

/*
 * view::AbstractView::unpackMouseCoordinates
 */
void view::AbstractView::unpackMouseCoordinates(float &x, float &y) {
    // intentionally empty
    // do something smart in the derived classes
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
    view::Camera_2::minimal_state_type minstate;
    this->_camera.get_minimal_state(minstate);
    this->_savedCameras[10].first = minstate;
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
    if (!this->_cameraSettingsSlot.Param<param::StringParam>()->Value().IsEmpty()) {
        std::string camstring(this->_cameraSettingsSlot.Param<param::StringParam>()->Value());
        cam_type::minimal_state_type minstate;
        if (!this->_cameraSerializer.deserialize(minstate, camstring)) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "The entered camera string was not valid. No change of the camera has been performed");
        } else {
            this->_camera = minstate;
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
    if (!this->GetCoreInstance()->IsmmconsoleFrontendCompatible()) {
        // new frontend
        const auto fit = std::find_if(this->frontend_resources.begin(), this->frontend_resources.end(),
            [](auto const& el) { return el.getIdentifier() == "LuaScriptPaths"; });

        if (fit != this->frontend_resources.end()) {
            core::utility::log::Log::DefaultLog.WriteInfo("[AbstractView] Got script paths");
            const auto& paths = fit->getResource<megamol::frontend_resources::ScriptPaths>().lua_script_paths;
            if (!paths.empty()) {
                path = paths[0];
            } else {
                return path;
            }
        } else {
            return path;
        }
    } else {
        path = this->GetCoreInstance()->GetLuaState()->GetScriptPath();
        if (path.empty())
            return path; // early exit for mmprj projects
    }
    const auto dotpos = path.find_last_of('.');
    path = path.substr(0, dotpos);
    path.append("_cam.json");
    return path;
}

/*
 * view::AbstractView::BkgndColour
 */
glm::vec4 view::AbstractView::BkgndColour(void) const {
    if (this->_bkgndColSlot.IsDirty()) {
        this->_bkgndColSlot.ResetDirty();
        this->_bkgndColSlot.Param<param::ColorParam>()->Value(this->_bkgndCol.r, this->_bkgndCol.g, this->_bkgndCol.b);
    }
    return this->_bkgndCol;
}
