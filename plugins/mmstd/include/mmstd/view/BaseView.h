/*
 * AbstractView.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BASEVIEW_H_INCLUDED
#define MEGAMOLCORE_BASEVIEW_H_INCLUDED

#include "mmcore/param/BoolParam.h"
#include "mmcore/view/AbstractView.h"
#include "mmstd/renderer/AbstractCallRenderView.h"

namespace megamol {
namespace core {
namespace view {

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
class BaseView : public ABSTRACTVIEW_TYPE {
public:
    explicit BaseView(AbstractView::ViewDimension dim);
    ~BaseView() = default;

    void beforeRender(double time, double instanceTime);

    void afterRender();

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    virtual bool GetExtents(Call& call) override;

    /**
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    virtual bool OnRenderView(Call& call) override;

    /**
     * Resets the view. This normally sets the camera parameters to
     * default values.
     */
    void ResetView();

protected:
    /**
     * Implementation of 'Release'.
     */
    virtual void release() {
        _fbo.reset();
    }

    /**
     * Sets camera aspect ratio if applicable.
     *
     * @param width The new width.
     * @param height The new height.
     */
    virtual void Resize(unsigned int width, unsigned int height) override;

    virtual bool OnKey(view::Key key, view::KeyAction action, view::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

protected:
    std::shared_ptr<typename VIEWCALL_TYPE::FBO_TYPE> _fbo;

    CAM_CONTROLLER_TYPE _camera_controller;
};

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::BaseView(AbstractView::ViewDimension dim)
        : ABSTRACTVIEW_TYPE(dim)
        , _fbo(nullptr)
        , _camera_controller(&this->_camera) {

    // none of the saved camera states are valid right now
    for (auto& e : this->_savedCameras) {
        e.second = false;
    }

    auto cam_ctrl_param_slots = this->_camera_controller.getParameterSlots();
    for (auto& param_slot : cam_ctrl_param_slots) {
        this->MakeSlotAvailable(param_slot);
    }
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline void BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::beforeRender(
    double time, double instanceTime) {
    AbstractView::beforeRender(time, instanceTime);

    // get camera values from params(?)
    _camera_controller.applyParameterSlotsToCamera(ABSTRACTVIEW_TYPE::_bboxs);

    // handle 3D view specific camera implementation
    float dt = std::chrono::duration<float>(this->_lastFrameDuration).count();
    _camera_controller.handleCameraMovement(dt);

    // set camera values to params
    _camera_controller.setParameterSlotsFromCamera();
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline void BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::afterRender() {
    AbstractView::afterRender();
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline bool BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::GetExtents(Call& call) {
    VIEWCALL_TYPE* crv = dynamic_cast<VIEWCALL_TYPE*>(&call);
    if (crv == nullptr) {
        return false;
    }

    AbstractCallRender* cr = this->_rhsRenderSlot.template CallAs<AbstractCallRender>();
    if (cr == nullptr) {
        return false;
    }
    cr->SetCamera(this->_camera);

    if (!(*cr)(AbstractCallRender::FnGetExtents)) {
        return false;
    }

    crv->SetTimeFramesCount(cr->TimeFramesCount());
    crv->SetIsInSituTime(cr->IsInSituTime());
    return true;
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline bool BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::OnRenderView(Call& call) {
    VIEWCALL_TYPE* crv = dynamic_cast<VIEWCALL_TYPE*>(&call);
    if (crv == NULL) {
        return false;
    }

    // get time from incoming call
    double time = crv->Time();
    if (time < 0.0f)
        time = this->DefaultTime(crv->InstanceTime());
    double instanceTime = crv->InstanceTime();

    auto fbo = _fbo;
    _fbo = crv->GetFramebuffer();

    auto cam_pose = this->_camera.template get<Camera::Pose>();
    auto cam_type = this->_camera.template get<Camera::ProjectionType>();
    if (cam_type == Camera::ORTHOGRAPHIC) {
        auto cam_intrinsics = this->_camera.template get<Camera::OrthographicParameters>();
        cam_intrinsics.aspect = static_cast<float>(_fbo->getWidth()) / static_cast<float>(_fbo->getHeight());
        this->_camera = Camera(cam_pose, cam_intrinsics);
    } else if (cam_type == Camera::PERSPECTIVE) {
        auto cam_intrinsics = this->_camera.template get<Camera::PerspectiveParameters>();
        cam_intrinsics.aspect = static_cast<float>(_fbo->getWidth()) / static_cast<float>(_fbo->getHeight());
        this->_camera = Camera(cam_pose, cam_intrinsics);
    }

    this->Render(time, instanceTime);

    _fbo = fbo;
    // only re-apply aspect ratio from copy, because otherwise camera updates handled within Render(...) are
    // lost
    cam_pose = this->_camera.template get<Camera::Pose>();
    cam_type = this->_camera.template get<Camera::ProjectionType>();
    if (cam_type == Camera::ORTHOGRAPHIC) {
        auto cam_intrinsics = this->_camera.template get<Camera::OrthographicParameters>();
        cam_intrinsics.aspect = static_cast<float>(_fbo->getWidth()) / static_cast<float>(_fbo->getHeight());
        this->_camera = Camera(cam_pose, cam_intrinsics);
    } else if (cam_type == Camera::PERSPECTIVE) {
        auto cam_intrinsics = this->_camera.template get<Camera::PerspectiveParameters>();
        cam_intrinsics.aspect = static_cast<float>(_fbo->getWidth()) / static_cast<float>(_fbo->getHeight());
        this->_camera = Camera(cam_pose, cam_intrinsics);
    }

    return true;
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline void BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::ResetView() {
    if (this->_cameraIsMutable) { // check if view is in control of the camera
        AbstractCallRender* cr = this->_rhsRenderSlot.template CallAs<AbstractCallRender>();
        if ((cr != nullptr) && (_fbo != nullptr) && ((*cr)(AbstractCallRender::FnGetExtents))) {
            this->_camera_controller.reset(
                this->_bboxs, static_cast<float>(_fbo->getWidth()) / static_cast<float>(_fbo->getHeight()));
        }
    } else {
        // TODO print warning
    }
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline void BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::Resize(
    unsigned int width, unsigned int height) {

    if (this->_cameraIsMutable) { // view seems to be in control of the camera
        auto cam_pose = this->_camera.template get<Camera::Pose>();
        if (this->_camera.template get<Camera::ProjectionType>() == Camera::ProjectionType::PERSPECTIVE) {
            auto cam_intrinsics = this->_camera.template get<Camera::PerspectiveParameters>();
            cam_intrinsics.aspect = static_cast<float>(width) / static_cast<float>(height);
            this->_camera = Camera(cam_pose, cam_intrinsics);
        } else if (this->_camera.template get<Camera::ProjectionType>() == Camera::ProjectionType::ORTHOGRAPHIC) {
            auto cam_intrinsics = this->_camera.template get<Camera::OrthographicParameters>();
            cam_intrinsics.aspect = static_cast<float>(width) / static_cast<float>(height);
            this->_camera = Camera(cam_pose, cam_intrinsics);
        }
    }
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline bool BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::OnKey(
    view::Key key, view::KeyAction action, view::Modifiers mods) {
    auto* cr = this->_rhsRenderSlot.template CallAs<AbstractCallRender>();
    if (cr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(AbstractCallRender::FnOnKey))
            return true;
    }

    if (this->_cameraIsMutable) {
        if (action == KeyAction::PRESS && (key >= Key::KEY_0 && key <= Key::KEY_9)) {
            int index =
                static_cast<int>(key) - static_cast<int>(Key::KEY_0); // ugly hack, maybe this can be done better
            index = (index - 1) % 10;                                 // put key '1' at index 0
            index = index < 0 ? index + 10 : index;                   // wrap key '0' to a positive index '9'

            if (mods.test(Modifier::CTRL)) {
                this->_savedCameras[index].first = this->_camera;
                this->_savedCameras[index].second = true;
                if (this->_autoSaveCamSettingsSlot.template Param<param::BoolParam>()->Value()) {
                    this->onStoreCamera(this->_storeCameraSettingsSlot); // manually trigger the storing
                }
            } else {
                if (this->_savedCameras[index].second) {
                    this->_camera = this->_savedCameras[index].first; // override current camera
                }
            }
        }

        this->_camera_controller.OnKey(key, action, mods);
    }

    return false;
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline bool BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::OnChar(unsigned int codePoint) {
    auto* cr = this->_rhsRenderSlot.template CallAs<AbstractCallRender>();
    if (cr == NULL)
        return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(AbstractCallRender::FnOnChar))
        return false;

    if (this->_cameraIsMutable) {
        this->_camera_controller.OnChar(codePoint);
    }

    return true;
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline bool BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::OnMouseButton(
    view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) {
    if (!this->_camera_controller.isOverriding() && !this->_camera_controller.isActive()) {
        auto* cr = this->_rhsRenderSlot.template CallAs<AbstractCallRender>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseButton;
            evt.mouseButtonData.button = button;
            evt.mouseButtonData.action = action;
            evt.mouseButtonData.mods = mods;
            cr->SetInputEvent(evt);
            if ((*cr)(AbstractCallRender::FnOnMouseButton))
                return true;
        }
    }

    if (this->_cameraIsMutable) {
        // get window resolution to help computing mouse coordinates
        int wndWidth;
        int wndHeight;
        auto projType = this->_camera.template get<Camera::ProjectionType>();
        if (projType == Camera::ProjectionType::PERSPECTIVE) {
            auto tile_end = this->_camera.template get<Camera::PerspectiveParameters>().image_plane_tile.tile_end;
            auto tile_start = this->_camera.template get<Camera::PerspectiveParameters>().image_plane_tile.tile_start;
            auto tile_size = tile_end - tile_start;

            wndWidth = static_cast<int>(static_cast<float>(this->_fbo->getWidth()) / tile_size.x);
            wndHeight = static_cast<int>(static_cast<float>(this->_fbo->getHeight()) / tile_size.y);
        } else if (projType == Camera::ProjectionType::ORTHOGRAPHIC) {
            auto tile_end = this->_camera.template get<Camera::OrthographicParameters>().image_plane_tile.tile_end;
            auto tile_start = this->_camera.template get<Camera::OrthographicParameters>().image_plane_tile.tile_start;
            auto tile_size = tile_end - tile_start;

            wndWidth = static_cast<int>(static_cast<float>(this->_fbo->getWidth()) / tile_size.x);
            wndHeight = static_cast<int>(static_cast<float>(this->_fbo->getHeight()) / tile_size.y);
        } else {
            return false; // Oh bother...
        }

        this->_camera_controller.OnMouseButton(button, action, mods, wndWidth, wndHeight);
    }

    return true;
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline bool BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::OnMouseMove(double x, double y) {
    if (!this->_camera_controller.isActive()) {
        auto* cr = this->_rhsRenderSlot.template CallAs<AbstractCallRender>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = x;
            evt.mouseMoveData.y = y;
            cr->SetInputEvent(evt);
            if ((*cr)(AbstractCallRender::FnOnMouseMove))
                return true;
        }
    }

    if (this->_cameraIsMutable) {
        // get window resolution to help computing mouse coordinates
        int wndWidth;
        int wndHeight;
        auto projType = this->_camera.template get<Camera::ProjectionType>();
        if (projType == Camera::ProjectionType::PERSPECTIVE) {
            auto tile_end = this->_camera.template get<Camera::PerspectiveParameters>().image_plane_tile.tile_end;
            auto tile_start = this->_camera.template get<Camera::PerspectiveParameters>().image_plane_tile.tile_start;
            auto tile_size = tile_end - tile_start;

            wndWidth = static_cast<int>(static_cast<float>(this->_fbo->getWidth()) / tile_size.x);
            wndHeight = static_cast<int>(static_cast<float>(this->_fbo->getHeight()) / tile_size.y);
        } else if (projType == Camera::ProjectionType::ORTHOGRAPHIC) {
            auto tile_end = this->_camera.template get<Camera::OrthographicParameters>().image_plane_tile.tile_end;
            auto tile_start = this->_camera.template get<Camera::OrthographicParameters>().image_plane_tile.tile_start;
            auto tile_size = tile_end - tile_start;

            wndWidth = static_cast<int>(static_cast<float>(this->_fbo->getWidth()) / tile_size.x);
            wndHeight = static_cast<int>(static_cast<float>(this->_fbo->getHeight()) / tile_size.y);
        } else {
            return false; // Oh bother...
        }

        this->_camera_controller.OnMouseMove(x, y, wndWidth, wndHeight);
    }

    return true;
}

template<typename VIEWCALL_TYPE, typename CAM_CONTROLLER_TYPE, typename ABSTRACTVIEW_TYPE>
inline bool BaseView<VIEWCALL_TYPE, CAM_CONTROLLER_TYPE, ABSTRACTVIEW_TYPE>::OnMouseScroll(double dx, double dy) {
    auto* cr = this->_rhsRenderSlot.template CallAs<view::AbstractCallRender>();
    if (cr != NULL) {
        view::InputEvent evt;
        evt.tag = view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        cr->SetInputEvent(evt);
        if ((*cr)(view::AbstractCallRender::FnOnMouseScroll))
            return true;
    }

    if (this->_cameraIsMutable) {
        this->_camera_controller.OnMouseScroll(dx, dy);
    }

    return true;
}

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_BASEVIEW_H_INCLUDED */
