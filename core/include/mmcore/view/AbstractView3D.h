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
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/Camera_2.h"
#include "mmcore/view/CallRender3D.h"

namespace megamol {
namespace core {
namespace view {

class MEGAMOLCORE_API AbstractView3D : public view::AbstractView {

public:

    /** Enum for default views from the respective direction */
    enum defaultview {
        DEFAULTVIEW_FRONT,
        DEFAULTVIEW_BACK,
        DEFAULTVIEW_RIGHT,
        DEFAULTVIEW_LEFT,
        DEFAULTVIEW_TOP,
        DEFAULTVIEW_BOTTOM,
    };

    /** Enum for default orientations from the respective direction */
    enum defaultorientation {
        DEFAULTORIENTATION_TOP,
        DEFAULTORIENTATION_RIGHT,
        DEFAULTORIENTATION_BOTTOM,
        DEFAULTORIENTATION_LEFT
    };

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

    bool mouseSensitivityChanged(param::ParamSlot& p);

    /**
     * Handles a request for the camera parameters used by the view.
     *
     * @param c The call being executed.
     *
     * @return true in case of success, false otherwise.
     */
    // virtual bool OnGetCamParams(view::CallCamParamSync& c);

    bool onToggleButton(param::ParamSlot& p);

    /*
     * Performs the actual camera movement based on the pressed keys
     */
    void handleCameraMovement(void);

#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
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

    /** Flag showing the look at point */
    param::ParamSlot _showLookAt;

    /** focus distance for stereo projection */
    param::ParamSlot _stereoFocusDistSlot;

    /** eye distance for stereo projection */
    param::ParamSlot _stereoEyeDistSlot;

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

    param::ParamSlot _hookOnChangeOnlySlot;

    /** Enable selecting mode of mouse (disables camera movement) */
    param::ParamSlot _enableMouseSelectionSlot;

    /** Invisible parameters for lua manipulation */
    param::ParamSlot _cameraPositionParam;
    param::ParamSlot _cameraOrientationParam;
    param::ParamSlot _cameraProjectionTypeParam;
    param::ParamSlot _cameraNearPlaneParam;
    param::ParamSlot _cameraFarPlaneParam;
    param::ParamSlot _cameraConvergencePlaneParam;
    param::ParamSlot _cameraEyeParam;
    param::ParamSlot _cameraGateScalingParam;
    param::ParamSlot _cameraFilmGateParam;
    param::ParamSlot _cameraResolutionXParam;
    param::ParamSlot _cameraResolutionYParam;
    param::ParamSlot _cameraCenterOffsetParam;
    param::ParamSlot _cameraHalfApertureDegreesParam;
    param::ParamSlot _cameraHalfDisparityParam;

    /** Camara override parameters */
    param::ParamSlot _cameraOvrUpParam;
    param::ParamSlot _cameraOvrLookatParam;
    param::ParamSlot _cameraOvrParam;

    /** Shows the view cube helper */
    param::ParamSlot _showViewCubeParam;

    /** Standard camera views */
    param::ParamSlot _cameraViewOrientationParam;
    param::ParamSlot _cameraSetViewChooserParam;
    param::ParamSlot _cameraSetOrientationChooserParam;

    bool cameraOvrCallback(param::ParamSlot& p);

    /** Map storing the pressed state of all keyboard buttons */
    std::map<view::Key, bool> _pressedKeyMap;

    /** Map storing the pressed state of all mouse buttons */
    std::map<view::MouseButton, bool> _pressedMouseMap;

    /** Flag determining whether the arcball is the default steering method of the camera */
    bool _arcballDefault;

    /** Center of rotation for orbital manipulators */
    glm::vec3 _rotCenter;

    /** Value storing whether there have been read parameter values that came from outside */
    bool _valuesFromOutside;

    bool _cameraControlOverrideActive;
};

} // namespace view
} /* end namespace core */
} /* end namespace megamol */

