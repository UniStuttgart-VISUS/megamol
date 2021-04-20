#ifndef CAMERA_CONTROLLER_H_INCLUDED
#define CAMERA_CONTROLLER_H_INCLUDED

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/cam_typedefs.h"
#include "mmcore/view/Camera.h"

#include "glm/gtx/rotate_vector.hpp" // glm::rotate(quaternion, vector)
#include "glm/gtx/quaternion.hpp" // glm::rotate(quat, vector)

namespace megamol {
namespace core {
    namespace view {

        class Camera3DController {
        public:
            Camera3DController(Camera* target_camera, bool arcball_is_default = false)
                    : _viewKeyMoveStepSlot("viewKey::MoveStep", "The move step size in world coordinates")
                    , _viewKeyRunFactorSlot(
                          "viewKey::RunFactor", "The factor for step size multiplication when running (shift)")
                    , _viewKeyAngleStepSlot("viewKey::AngleStep", "The angle rotate step in degrees")
                    , _viewKeyFixToWorldUpSlot("viewKey::FixToWorldUp", "Fix rotation manipulator to world up vector")
                    , _mouseSensitivitySlot("viewKey::MouseSensitivity", "used for WASD mode")
                    , _viewKeyRotPointSlot("viewKey::RotPoint", "The point around which the view will be rotated")
                    , _target_camera(target_camera)
                    , _cameraControlOverrideActive(false)
                    , _arcballDefault(false) {

                this->_viewKeyMoveStepSlot.SetParameter(new param::FloatParam(0.5f, 0.001f));

                this->_viewKeyRunFactorSlot.SetParameter(new param::FloatParam(2.0f, 0.1f));

                this->_viewKeyAngleStepSlot.SetParameter(new param::FloatParam(90.0f, 0.1f, 360.0f));

                this->_viewKeyFixToWorldUpSlot.SetParameter(new param::BoolParam(true));

                this->_mouseSensitivitySlot.SetParameter(new param::FloatParam(3.0f, 0.001f, 10.0f));

                // TODO clean up vrpsev memory after use
                param::EnumParam* vrpsev = new param::EnumParam(1);
                vrpsev->SetTypePair(0, "Position");
                vrpsev->SetTypePair(1, "Look-At");
                this->_viewKeyRotPointSlot.SetParameter(vrpsev);

                _translateManipulator.set_target(_target_camera);
                _translateManipulator.enable();

                _rotateManipulator.set_target(_target_camera);
                _rotateManipulator.enable();

                _arcballManipulator.set_target(_target_camera);
                _arcballManipulator.enable();
                _rotCenter = glm::vec3(0.0f, 0.0f, 0.0f);

                _turntableManipulator.set_target(_target_camera);
                _turntableManipulator.enable();

                _orbitAltitudeManipulator.set_target(_target_camera);
                _orbitAltitudeManipulator.enable();
            }
            ~Camera3DController() {}

            void setTargetCamera(Camera* target_camera) {
                _target_camera = target_camera;

                _translateManipulator.set_target(_target_camera);

                _rotateManipulator.set_target(_target_camera);

                _arcballManipulator.set_target(_target_camera);

                _turntableManipulator.set_target(_target_camera);

                _orbitAltitudeManipulator.set_target(_target_camera);
            }

            void setRotationalCenter(glm::vec3 new_center) {
                _rotCenter = new_center;
            }

            glm::vec3 getRotationalCenter() {
                return _rotCenter;
            }

            void setArcballDefault(bool default_to_arcball) {
                _arcballDefault = default_to_arcball;
            }

            std::vector<AbstractSlot*> getParameterSlots() {
                return {&this->_viewKeyMoveStepSlot, &this->_viewKeyRunFactorSlot, &this->_viewKeyAngleStepSlot,
                    &this->_viewKeyFixToWorldUpSlot, &this->_mouseSensitivitySlot, &this->_viewKeyRotPointSlot};
            }

            bool isActive() {
                return _arcballManipulator.manipulating() || _translateManipulator.manipulating() ||
                       _rotateManipulator.manipulating() || _turntableManipulator.manipulating() ||
                       _orbitAltitudeManipulator.manipulating();
            }

            bool isOverriding() {
                return _cameraControlOverrideActive;
            }

            /*
             * Performs the actual camera movement based on the pressed keys
             */
            void handleCameraMovement(float dt) {
                float step = this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value();
                step *= dt;

                const float runFactor = this->_viewKeyRunFactorSlot.Param<param::FloatParam>()->Value();
                if (this->_modkeys.test(Modifier::SHIFT)) {
                    step *= runFactor;
                }

                bool anymodpressed = !this->_modkeys.none();
                float rotationStep = this->_viewKeyAngleStepSlot.Param<param::FloatParam>()->Value();
                rotationStep *= dt;

                glm::vec3 currCamPos(this->_target_camera->get<Camera::Pose>().position);
                float orbitalAltitude = glm::length(currCamPos - _rotCenter);

                if (this->_translateManipulator.manipulating()) {

                    if (this->_pressedKeyMap.count(Key::KEY_W) > 0 && this->_pressedKeyMap[Key::KEY_W]) {
                        this->_translateManipulator.move_forward(step);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_S) > 0 && this->_pressedKeyMap[Key::KEY_S]) {
                        this->_translateManipulator.move_forward(-step);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_A) > 0 && this->_pressedKeyMap[Key::KEY_A]) {
                        this->_translateManipulator.move_horizontally(-step);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_D) > 0 && this->_pressedKeyMap[Key::KEY_D]) {
                        this->_translateManipulator.move_horizontally(step);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_C) > 0 && this->_pressedKeyMap[Key::KEY_C]) {
                        this->_translateManipulator.move_vertically(step);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_V) > 0 && this->_pressedKeyMap[Key::KEY_V]) {
                        this->_translateManipulator.move_vertically(-step);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_Q) > 0 && this->_pressedKeyMap[Key::KEY_Q]) {
                        this->_rotateManipulator.roll(-rotationStep);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_E) > 0 && this->_pressedKeyMap[Key::KEY_E]) {
                        this->_rotateManipulator.roll(rotationStep);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_UP) > 0 && this->_pressedKeyMap[Key::KEY_UP]) {
                        this->_rotateManipulator.pitch(-rotationStep);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_DOWN) > 0 && this->_pressedKeyMap[Key::KEY_DOWN]) {
                        this->_rotateManipulator.pitch(rotationStep);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_LEFT) > 0 && this->_pressedKeyMap[Key::KEY_LEFT]) {
                        this->_rotateManipulator.yaw(rotationStep);
                    }
                    if (this->_pressedKeyMap.count(Key::KEY_RIGHT) > 0 && this->_pressedKeyMap[Key::KEY_RIGHT]) {
                        this->_rotateManipulator.yaw(-rotationStep);
                    }
                }

                glm::vec3 newCamPos(this->_target_camera->get<Camera::Pose>().position);
                glm::vec3 camDir(this->_target_camera->get<Camera::Pose>().direction);
                _rotCenter = newCamPos + orbitalAltitude * glm::normalize(camDir);
            }

            /**
             * This event handler can be reimplemented to receive key code events.
             */
            virtual void OnKey(Key key, KeyAction action, Modifiers mods) {
                if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
                    this->_pressedKeyMap[key] = true;
                } else if (action == view::KeyAction::RELEASE) {
                    this->_pressedKeyMap[key] = false;
                }

                if (key == view::Key::KEY_LEFT_ALT || key == view::Key::KEY_RIGHT_ALT) {
                    if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
                        _modkeys.set(view::Modifier::ALT);
                        _cameraControlOverrideActive = true;
                    } else if (action == view::KeyAction::RELEASE) {
                        _modkeys.reset(view::Modifier::ALT);
                        _cameraControlOverrideActive = false;
                    }
                }
                if (key == view::Key::KEY_LEFT_SHIFT || key == view::Key::KEY_RIGHT_SHIFT) {
                    if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
                        _modkeys.set(view::Modifier::SHIFT);
                    } else if (action == view::KeyAction::RELEASE) {
                        _modkeys.reset(view::Modifier::SHIFT);
                    }
                }
                if (key == view::Key::KEY_LEFT_CONTROL || key == view::Key::KEY_RIGHT_CONTROL) {
                    if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
                        _modkeys.set(view::Modifier::CTRL);
                        _cameraControlOverrideActive = true;
                    } else if (action == view::KeyAction::RELEASE) {
                        _modkeys.reset(view::Modifier::CTRL);
                        _cameraControlOverrideActive = false;
                    }
                }
            }

            /**
             * This event handler can be reimplemented to receive unicode events.
             */
            virtual void OnChar(unsigned int codePoint) {}

            /**
             * This event handler can be reimplemented to receive mouse button events.
             */
            virtual void OnMouseButton(
                MouseButton button, MouseButtonAction action, Modifiers mods, int wndWidth, int wndHeight) {
                if (action == view::MouseButtonAction::PRESS) {
                    this->_pressedMouseMap[button] = true;
                } else if (action == view::MouseButtonAction::RELEASE) {
                    this->_pressedMouseMap[button] = false;
                }

                // This mouse handling/mapping is so utterly weird and should die!
                auto down = action == view::MouseButtonAction::PRESS;
                bool altPressed = mods.test(view::Modifier::ALT);   // _modkeys.test(view::Modifier::ALT);
                bool ctrlPressed = mods.test(view::Modifier::CTRL); // _modkeys.test(view::Modifier::CTRL);

                switch (button) {
                case megamol::core::view::MouseButton::BUTTON_LEFT:
                    if (!isActive()) {
                        if (altPressed ^
                            (this->_arcballDefault && !ctrlPressed)) // Left mouse press + alt/arcDefault+noCtrl ->
                                                                     // activate arcball manipluator
                        {
                            this->_arcballManipulator.setActive(
                                wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
                        } else if (ctrlPressed) // Left mouse press + Ctrl -> activate orbital manipluator
                        {
                            this->_turntableManipulator.setActive(
                                wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
                        }
                    }

                    break;
                case megamol::core::view::MouseButton::BUTTON_RIGHT:
                    if (!isActive()) {
                        if ((altPressed ^ this->_arcballDefault) || ctrlPressed) {
                            this->_orbitAltitudeManipulator.setActive(
                                wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
                        } else {
                            this->_rotateManipulator.setActive();
                            this->_translateManipulator.setActive(
                                wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
                        }
                    }

                    break;
                case megamol::core::view::MouseButton::BUTTON_MIDDLE:
                    if (!isActive()) {
                        this->_translateManipulator.setActive(
                            wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
                    }

                    break;
                default:
                    break;
                }

                if (action == view::MouseButtonAction::RELEASE) // Mouse release + no other mouse button pressed ->
                                                                // deactivate all mouse manipulators
                {
                    if (!(_pressedMouseMap[MouseButton::BUTTON_LEFT] || _pressedMouseMap[MouseButton::BUTTON_MIDDLE] ||
                            _pressedMouseMap[MouseButton::BUTTON_RIGHT])) {
                        this->_arcballManipulator.setInactive();
                        this->_orbitAltitudeManipulator.setInactive();
                        this->_rotateManipulator.setInactive();
                        this->_turntableManipulator.setInactive();
                        this->_translateManipulator.setInactive();
                    }
                }
            }

            /**
             * This event handler can be reimplemented to receive mouse move events.
             */
            virtual void OnMouseMove(double x, double y, int wndWidth, int wndHeight) {
                this->_mouseX = (float) static_cast<int>(x);
                this->_mouseY = (float) static_cast<int>(y);

                glm::vec3 newPos;

                if (this->_turntableManipulator.manipulating()) {
                    this->_turntableManipulator.on_drag(wndWidth - static_cast<int>(this->_mouseX),
                        static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0), wndWidth, wndHeight);
                }

                if (this->_arcballManipulator.manipulating()) {
                    this->_arcballManipulator.on_drag(wndWidth - static_cast<int>(this->_mouseX),
                        static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0));
                }

                if (this->_orbitAltitudeManipulator.manipulating()) {
                    this->_orbitAltitudeManipulator.on_drag(wndWidth - static_cast<int>(this->_mouseX),
                        static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0));
                }

                if (this->_translateManipulator.manipulating() && !this->_rotateManipulator.manipulating()) {

                    // compute proper step size by computing pixel world size at distance to rotCenter
                    auto projType = _target_camera->get<Camera::ProjectionType>();
                    glm::vec3 currCamPos = this->_target_camera->get<Camera::Pose>().position;
                    float orbitalAltitude = glm::length(currCamPos - _rotCenter);
                    float pixel_world_size;
                    if (projType == Camera::ProjectionType::PERSPECTIVE) {
                        auto fovy = _target_camera->get<Camera::PerspectiveParameters>().fovy;
                        auto vertical_height = std::tan(fovy) * orbitalAltitude;
                        pixel_world_size = vertical_height / static_cast<float>(wndHeight);
                    } else if (projType == Camera::ProjectionType::ORTHOGRAPHIC) {
                        auto vertical_height = _target_camera->get<Camera::OrthographicParameters>().frustrum_height;
                        pixel_world_size = vertical_height / static_cast<float>(wndHeight);
                    } else {
                        // Oh bother...
                    }

                    this->_translateManipulator.set_step_size(pixel_world_size);

                    this->_translateManipulator.move_horizontally(wndWidth - static_cast<int>(this->_mouseX));
                    this->_translateManipulator.move_vertically(static_cast<int>(this->_mouseY));
                }
            }

            /**
             * This event handler can be reimplemented to receive mouse scroll events.
             */
            virtual void OnMouseScroll(double dx, double dy) {
                // This mouse handling/mapping is so utterly weird and should die!
                if ((abs(dy) > 0.0)) {
                    if (this->_rotateManipulator.manipulating()) {
                        this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->SetValue(
                            this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value() +
                            (dy * 0.1f * this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value()));
                    } else {
                        auto cam_pose = _target_camera->get<Camera::Pose>();
                        auto v = glm::normalize(_rotCenter - cam_pose.position);
                        auto altitude = glm::length(_rotCenter - cam_pose.position);
                        cam_pose.position = cam_pose.position + (v * static_cast<float>(dy) * (altitude / 50.0f));
                        _target_camera->setPose(cam_pose);
                    }
                }
            }

        private:
            /** The move step size in world coordinates */
            param::ParamSlot _viewKeyMoveStepSlot;

            param::ParamSlot _viewKeyRunFactorSlot;

            /** The angle rotate step in degrees */
            param::ParamSlot _viewKeyAngleStepSlot;

            param::ParamSlot _viewKeyFixToWorldUpSlot;

            /** sensitivity for mouse rotation in WASD mode */
            param::ParamSlot _mouseSensitivitySlot;

            /** The point around which the view will be roateted */
            param::ParamSlot _viewKeyRotPointSlot;


            //////
            // Individual manipulators
            //////
            /** The orbital arcball manipulator for the camera */
            arcball_type _arcballManipulator;

            /** The translation manipulator for the camera */
            xlate_type _translateManipulator;

            /** The rotation manipulator for the camera */
            rotate_type _rotateManipulator;

            /** The orbital manipulator turntable for the camera */
            turntable_type _turntableManipulator;

            /** The manipulator for changing the orbital altitude */
            orbit_altitude_type _orbitAltitudeManipulator;

            /** Center of rotation for orbital manipulators */
            glm::vec3 _rotCenter;

            /** Flag determining whether the arcball is the default steering method of the camera */
            bool _arcballDefault;


            //////
            // Track controller state...
            //////

            bool _cameraControlOverrideActive;

            /** The camera that is currently controlled */
            Camera* _target_camera; // TODO not liking this...

            //////
            // Mirror mouse and keyboard state...
            //////

            /** The mouse x coordinate */
            float _mouseX;

            /** The mouse y coordinate */
            float _mouseY;

            /** Map storing the pressed state of all keyboard buttons */
            std::map<view::Key, bool> _pressedKeyMap;

            /** the set input modifiers*/
            core::view::Modifiers _modkeys;

            /** Map storing the pressed state of all mouse buttons */
            std::map<view::MouseButton, bool> _pressedMouseMap;
        };

        class Camera2DController {
        public:
            Camera2DController(Camera* target_camera) : _target_camera(target_camera) {}
            ~Camera2DController(){}

            std::vector<AbstractSlot*> getParameterSlots() {
                return {};
            }

            bool isActive() {
                return false; //TODO
            }

            bool isOverriding() {
                return false; //TODO
            }

            /**
             * This event handler can be reimplemented to receive key code events.
             */
            virtual void OnKey(Key key, KeyAction action, Modifiers mods) {
                _ctrlDown = mods.test(core::view::Modifier::CTRL);
            }

            /**
             * This event handler can be reimplemented to receive unicode events.
             */
            virtual void OnChar(unsigned int codePoint) {
            }

            /**
             * This event handler can be reimplemented to receive mouse button events.
             */
            virtual void OnMouseButton(
                MouseButton button, MouseButtonAction action, Modifiers mods, int wndWidth, int wndHeight) {
                this->_mouseMode = MouseMode::Propagate;

                if(_ctrlDown) {
                    auto down = action == MouseButtonAction::PRESS;
                    if (button == MouseButton::BUTTON_LEFT && down) {
                        this->_mouseMode = MouseMode::Pan;
                    } else if (button == MouseButton::BUTTON_MIDDLE && down) {
                        this->_mouseMode = MouseMode::Zoom;
                    }
                }
                else {
                    if (button == MouseButton::BUTTON_MIDDLE && action == MouseButtonAction::PRESS) {
                        this->_mouseMode = MouseMode::Pan;
                    }
                }
            }

            /**
             * This event handler can be reimplemented to receive mouse move events.
             */
            virtual void OnMouseMove(double x, double y, int wndWidth, int wndHeight) {
                if (this->_mouseMode == MouseMode::Pan) {

                        // compute size of a pixel in world space
                        float stepSize =
                            _target_camera->get<Camera::OrthographicParameters>().frustrum_height / wndHeight;
                        auto dx = (this->_mouseX - x) * stepSize;
                        auto dy = (this->_mouseY - y) * stepSize;

                        auto cam_pose = _target_camera->get<Camera::Pose>();
                        cam_pose.position += glm::vec3(dx, -dy, 0.0f);

                        _target_camera->setPose(cam_pose);


                } else if (this->_mouseMode == MouseMode::Zoom) {

                        auto dy = (this->_mouseY - y);

                        auto cam_pose = _target_camera->get<Camera::Pose>();
                        auto cam_intrinsics = _target_camera->get<Camera::OrthographicParameters>();

                        cam_intrinsics.frustrum_height -= (dy / wndHeight) * (cam_intrinsics.frustrum_height);

                        *_target_camera = Camera(cam_pose, cam_intrinsics);
                }

                this->_mouseX = x;
                this->_mouseY = y;
            }

             /**
             * This event handler can be reimplemented to receive mouse scroll events.
             */
            virtual void OnMouseScroll(double dx, double dy) {
                auto cam_pose = _target_camera->get<Camera::Pose>();
                auto cam_intrinsics = _target_camera->get<Camera::OrthographicParameters>();
                cam_intrinsics.frustrum_height -= (dy / 10.0) * (cam_intrinsics.frustrum_height);

                *_target_camera = Camera(cam_pose, cam_intrinsics);
            }

            enum MouseMode : uint8_t { Propagate, Pan, Zoom };

        private:
            /** The camera that is currently controlled */
            Camera* _target_camera; // TODO not liking this...

            /** Track state of ctrl key for camera controls */
            bool _ctrlDown;

            /** The mouse drag mode */
            MouseMode _mouseMode;

            /** The mouse x coordinate */
            float _mouseX;

            /** The mouse y coordinate */
            float _mouseY;
        };

    } // namespace view
} // namespace core
} // namespace megamol

#endif // !CAMERA_CONTROLLER_H_INCLUDED
