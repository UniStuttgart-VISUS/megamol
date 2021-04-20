/*
 * AbstractView.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BASEVIEW_H_INCLUDED
#define MEGAMOLCORE_BASEVIEW_H_INCLUDED

#include "AbstractView.h"

namespace megamol {
namespace core {
    namespace view {

        template<typename FBO_TYPE>
        using RESIZEFUNC = void(std::shared_ptr<FBO_TYPE>&, int, int);

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        class MEGAMOLCORE_API BaseView : public AbstractView {
        public:
            BaseView();
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

            
            /**
             * Callback requesting a rendering of this view
             *
             * @param call The calling call
             *
             * @return The return value
             */
            virtual bool OnRenderView(Call& call) override;

        protected:
            /**
             * Implementation of 'Create'.
             *
             * @return 'true' on success, 'false' otherwise.
             */
            virtual bool create(void);

            /**
             * Implementation of 'Release'.
             */
            virtual void release() {
                _fbo.reset();
            }

            /**
             * Answer the camera synchronization number.
             *
             * @return The camera synchronization number
             */
            virtual unsigned int GetCameraSyncNumber(void) const {
                // implemented here to to avoid duplicated empty implementations in View3D(GL)
                return 0;
            }

            /**
             * Resizes the framebuffer object and sets camera aspect ratio if applicable.
             *
             * @param width The new width.
             * @param height The new height.
             */
            virtual void Resize(unsigned int width, unsigned int height) override;

            virtual bool OnKey(view::Key key, view::KeyAction action, view::Modifiers mods) override;

            virtual bool OnChar(unsigned int codePoint) override;

            virtual bool OnMouseButton(
                view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) override;

            virtual bool OnMouseMove(double x, double y) override;

            virtual bool OnMouseScroll(double dx, double dy) override;

        protected:
            std::shared_ptr<typename FBO_TYPE> _fbo;

            CAM_CONTROLLER_TYPE _camera_controller;

            CAM_PARAMS_TYPE _camera_parameters;
        };

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::BaseView()
                : _fbo(nullptr), _camera_controller(&this->_camera), _camera_parameters() {

            // none of the saved camera states are valid right now
            for (auto& e : this->_savedCameras) {
                e.second = false;
            }

            auto cam_ctrl_param_slots = this->_camera_controller.getParameterSlots();
            for (auto& param_slot : cam_ctrl_param_slots) {
                this->MakeSlotAvailable(param_slot);
            }

            auto cam_param_slots = this->_camera_parameters.getParameterSlots();
            for (auto& param_slot : cam_param_slots) {
                this->MakeSlotAvailable(param_slot);
            }
        }

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline void BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::beforeRender(
            double time, double instanceTime) {
            AbstractView::beforeRender(time, instanceTime);

            // get camera values from params(?)
            this->_camera = this->_camera_parameters.getCameraFromParameters(
                this->_camera, this->_camera_controller.getRotationalCenter());

            // handle 3D view specific camera implementation
            float dt = std::chrono::duration<float>(this->_lastFrameDuration).count();
            this->_camera_controller.handleCameraMovement(dt);

            // set camera values to params
            this->_camera_parameters.setParametersFromCamera(this->_camera);
        }

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline void BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::afterRender() {
            AbstractView::afterRender();
        }

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline bool BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::OnRenderView(Call& call) {
            return false;
        }

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline bool BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::create(void) {
            return false;
        }

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline void BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::Resize(
            unsigned int width, unsigned int height) {

            resize_func(_fbo, width, height);

            if (_cameraIsMutable) { // view seems to be in control of the camera
                auto cam_pose = _camera.get<Camera::Pose>();
                if (_camera.get<Camera::ProjectionType>() == Camera::ProjectionType::PERSPECTIVE) {
                    auto cam_intrinsics = _camera.get<Camera::PerspectiveParameters>();
                    cam_intrinsics.aspect = static_cast<float>(width) / static_cast<float>(height);
                    _camera = Camera(cam_pose, cam_intrinsics);
                } else if (_camera.get<Camera::ProjectionType>() == Camera::ProjectionType::ORTHOGRAPHIC) {
                    auto cam_intrinsics = _camera.get<Camera::OrthographicParameters>();
                    cam_intrinsics.aspect = static_cast<float>(width) / static_cast<float>(height);
                    _camera = Camera(cam_pose, cam_intrinsics);
                }
            }
        }

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline bool BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::OnKey(
            view::Key key, view::KeyAction action, view::Modifiers mods) {
            auto* cr = this->_rhsRenderSlot.CallAs<AbstractCallRender>();
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
                    int index = static_cast<int>(key) -
                                static_cast<int>(Key::KEY_0); // ugly hack, maybe this can be done better
                    index = (index - 1) % 10;                 // put key '1' at index 0
                    index = index < 0 ? index + 10 : index;   // wrap key '0' to a positive index '9'

                    if (mods.test(Modifier::CTRL)) {
                        this->_savedCameras[index].first = this->_camera;
                        this->_savedCameras[index].second = true;
                        if (this->_autoSaveCamSettingsSlot.Param<param::BoolParam>()->Value()) {
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

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline bool BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::OnChar(
            unsigned int codePoint) {
            auto* cr = this->_rhsRenderSlot.CallAs<AbstractCallRender>();
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

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline bool BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::OnMouseButton(
            view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) {
            if (!this->_camera_controller.isOverriding() && !this->_camera_controller.isActive()) {
                auto* cr = this->_rhsRenderSlot.CallAs<AbstractCallRender>();
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
                auto projType = this->_camera.get<Camera::ProjectionType>();
                if (projType == Camera::ProjectionType::PERSPECTIVE) {
                    auto tile_end = this->_camera.get<Camera::PerspectiveParameters>().image_plane_tile.tile_end;
                    auto tile_start = this->_camera.get<Camera::PerspectiveParameters>().image_plane_tile.tile_start;
                    auto tile_size = tile_end - tile_start;

                    wndWidth = static_cast<int>(static_cast<float>(this->_fbo->getWidth()) / tile_size.x);
                    wndHeight = static_cast<int>(static_cast<float>(this->_fbo->getHeight()) / tile_size.y);
                } else if (projType == Camera::ProjectionType::ORTHOGRAPHIC) {
                    auto tile_end = this->_camera.get<Camera::OrthographicParameters>().image_plane_tile.tile_end;
                    auto tile_start = this->_camera.get<Camera::OrthographicParameters>().image_plane_tile.tile_start;
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

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline bool BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::OnMouseMove(
            double x, double y) {
            if (!this->_camera_controller.isActive()) {
                auto* cr = this->_rhsRenderSlot.CallAs<AbstractCallRender>();
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
                auto projType = this->_camera.get<Camera::ProjectionType>();
                if (projType == Camera::ProjectionType::PERSPECTIVE) {
                    auto tile_end = this->_camera.get<Camera::PerspectiveParameters>().image_plane_tile.tile_end;
                    auto tile_start = this->_camera.get<Camera::PerspectiveParameters>().image_plane_tile.tile_start;
                    auto tile_size = tile_end - tile_start;

                    wndWidth = static_cast<int>(static_cast<float>(this->_fbo->getWidth()) / tile_size.x);
                    wndHeight = static_cast<int>(static_cast<float>(this->_fbo->getHeight()) / tile_size.y);
                } else if (projType == Camera::ProjectionType::ORTHOGRAPHIC) {
                    auto tile_end = this->_camera.get<Camera::OrthographicParameters>().image_plane_tile.tile_end;
                    auto tile_start = this->_camera.get<Camera::OrthographicParameters>().image_plane_tile.tile_start;
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

        template<typename FBO_TYPE, RESIZEFUNC<typename FBO_TYPE> resize_func, typename CAM_CONTROLLER_TYPE,
            typename CAM_PARAMS_TYPE>
        inline bool BaseView<FBO_TYPE, resize_func, CAM_CONTROLLER_TYPE, CAM_PARAMS_TYPE>::OnMouseScroll(
            double dx, double dy) {
            auto* cr = this->_rhsRenderSlot.CallAs<view::AbstractCallRender>();
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
