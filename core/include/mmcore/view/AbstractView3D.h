/*
 * AbstractView3D.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <map>
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractRenderingView.h"
#include "mmcore/view/Camera_2.h"
#include "mmcore/view/TimeControl.h"
#include "mmcore/view/CallRender3D.h"

namespace megamol {
namespace core {
namespace view {

class MEGAMOLCORE_API AbstractView3D : public view::AbstractRenderingView {

public:


    /**
     * Answer the default time for this view
     *
     * @param instTime the current instance time
     *
     * @return The default time
     */
    virtual float DefaultTime(double instTime) const { return this->timeCtrl.Time(instTime); }

    /**
     * Answer the camera synchronization number.
     *
     * @return The camera synchronization number
     */
    virtual unsigned int GetCameraSyncNumber(void) const;

    void beforeRender(const mmcRenderViewContext& context);

    void afterRender(const mmcRenderViewContext& context);

    /**
     * Resets the view. This normally sets the camera parameters to
     * default values.
     */
    virtual void ResetView(void);

    /**
     * Resizes the AbstractAbstractView3D.
     *
     * @param width The new width.
     * @param height The new height.
     */
    virtual void Resize(unsigned int width, unsigned int height);

    /**
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    virtual bool OnRenderView(Call& call);

    /**
     * Freezes, updates, or unfreezes the view onto the scene (not the
     * rendering, but camera settings, timing, etc).
     *
     * @param freeze true means freeze or update freezed settings,
     *               false means unfreeze
     */
    virtual void UpdateFreeze(bool freeze);

    virtual bool OnKey(view::Key key, view::KeyAction action, view::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

protected:
    /** Ctor. */
    AbstractView3D(void);

    /** Dtor. */
    virtual ~AbstractView3D(void);


    /**
     * Unpacks the mouse coordinates, which are relative to the virtual
     * viewport size.
     *
     * @param x The x coordinate of the mouse position
     * @param y The y coordinate of the mouse position
     */
    virtual void unpackMouseCoordinates(float& x, float& y);

    /**
     * Sets all parameters to the currently used camera values
     *
     * @param cam Camera containing the values that will be set
     */
    void setCameraValues(const core::view::Camera_2& cam);

    /**
     * Adapts camera values set by the user if necessary
     *
     * @param cam The camera the newly set parameters will be stored in
     * @return True if a camera value had to be adapted, false otherwise
     */
    bool adaptCameraValues(core::view::Camera_2& cam);

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    // protected variables //

#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** the set input modifiers*/
    core::view::Modifiers modkeys;
#ifdef _WIN32
#    pragma warning(default : 4251)
#endif /* _WIN32 */

    /** The complete scene bounding box */
    BoundingBoxes_2 bboxs;

    bool frameIsNew = false;

    bool mouseSensitivityChanged(param::ParamSlot& p);

    /**
     * Handles a request for the camera parameters used by the view.
     *
     * @param c The call being executed.
     *
     * @return true in case of success, false otherwise.
     */
    // virtual bool OnGetCamParams(view::CallCamParamSync& c);

    /**
     * Resets the view
     *
     * @param p Must be resetViewSlot
     *
     * @return true
     */
    bool onResetView(param::ParamSlot& p);

    bool onToggleButton(param::ParamSlot& p);

    /*
     * Performs the actual camera movement based on the pressed keys
     */
    void handleCameraMovement(void);

#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The orbital arcball manipulator for the camera */
    arcball_type arcballManipulator;

    /** The translation manipulator for the camera */
    xlate_type translateManipulator;

    /** The rotation manipulator for the camera */
    rotate_type rotateManipulator;

    /** The orbital manipulator turntable for the camera */
    turntable_type turntableManipulator;

    /** The manipulator for changing the orbital altitude */
    orbit_altitude_type orbitAltitudeManipulator;

    /** Slot to call the renderer to render */
    CallerSlot rendererSlot;

    /** Flag showing the look at point */
    param::ParamSlot showLookAt;

    /** Triggers the reset of the view */
    param::ParamSlot resetViewSlot;

    /** The incoming call */
    view::AbstractCallRender* overrideCall;

    /**
     * Flag if this is the first time an image gets created. Used for
     * initial camera reset
     */
    bool firstImg;

    /** focus distance for stereo projection */
    param::ParamSlot stereoFocusDistSlot;

    /** eye distance for stereo projection */
    param::ParamSlot stereoEyeDistSlot;

    /** The move step size in world coordinates */
    param::ParamSlot viewKeyMoveStepSlot;

    param::ParamSlot viewKeyRunFactorSlot;

    /** The angle rotate step in degrees */
    param::ParamSlot viewKeyAngleStepSlot;

    param::ParamSlot viewKeyFixToWorldUpSlot;

    /** sensitivity for mouse rotation in WASD mode */
    param::ParamSlot mouseSensitivitySlot;

    /** The point around which the view will be roateted */
    param::ParamSlot viewKeyRotPointSlot;

    param::ParamSlot hookOnChangeOnlySlot;

    /** Enable selecting mode of mouse (disables camera movement) */
    param::ParamSlot enableMouseSelectionSlot;

    /** Shows the view cube helper */
    param::ParamSlot showViewCubeSlot;

    /** whether to reset the view when the object bounding box changes */
    param::ParamSlot resetViewOnBBoxChangeSlot;

    /** Invisible parameters for lua manipulation */
    param::ParamSlot cameraPositionParam;
    param::ParamSlot cameraOrientationParam;
    param::ParamSlot cameraProjectionTypeParam;
    param::ParamSlot cameraNearPlaneParam;
    param::ParamSlot cameraFarPlaneParam;
    param::ParamSlot cameraConvergencePlaneParam;
    param::ParamSlot cameraEyeParam;
    param::ParamSlot cameraGateScalingParam;
    param::ParamSlot cameraFilmGateParam;
    param::ParamSlot cameraResolutionXParam;
    param::ParamSlot cameraResolutionYParam;
    param::ParamSlot cameraCenterOffsetParam;
    param::ParamSlot cameraHalfApertureDegreesParam;
    param::ParamSlot cameraHalfDisparityParam;

    /** Camara override parameters */
    param::ParamSlot cameraOvrUpParam;
    param::ParamSlot cameraOvrLookatParam;
    param::ParamSlot cameraOvrParam;

    bool cameraOvrCallback(param::ParamSlot& p);

    /** The time control */
    view::TimeControl timeCtrl;

    /** Map storing the pressed state of all keyboard buttons */
    std::map<view::Key, bool> pressedKeyMap;

    /** Map storing the pressed state of all mouse buttons */
    std::map<view::MouseButton, bool> pressedMouseMap;

    /** Flag determining whether the arcball is the default steering method of the camera */
    bool arcballDefault;

    /** Center of rotation for orbital manipulators */
    glm::vec3 rotCenter;

    /** Value storing whether there have been read parameter values that came from outside */
    bool valuesFromOutside;

    /**  */
    std::chrono::time_point<std::chrono::high_resolution_clock> lastFrameTime;

    std::chrono::microseconds lastFrameDuration;

    bool cameraControlOverrideActive;
};

} // namespace view
} /* end namespace core */
} /* end namespace megamol */

