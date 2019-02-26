/*
 * View3D_2.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_View3D_2_H_INCLUDED
#define MEGAMOLCORE_View3D_2_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <memory>
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/nextgen/Camera_2.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/AbstractCamParamSync.h"
#include "mmcore/view/AbstractRenderingView.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/TimeControl.h"
#include "vislib/graphics/CameraLookAtDist.h"
#include "vislib/graphics/CameraMove2D.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/graphics/CameraRotate2D.h"
#include "vislib/graphics/CameraRotate2DLookAt.h"
#include "vislib/graphics/CameraZoom2DAngle.h"
#include "vislib/graphics/CameraZoom2DMove.h"
#include "vislib/graphics/Cursor2D.h"
#include "vislib/graphics/InputModifiers.h"
#include "vislib/graphics/gl/CameraOpenGL.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/graphicstypes.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/PerformanceCounter.h"

namespace megamol {
namespace core {
namespace nextgen {

class MEGAMOLCORE_API View3D_2 : public view::AbstractRenderingView /*, public view::AbstractCamParamSync*/ {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "View3D_2"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "New and improved 3D View Module"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    View3D_2(void);

    /** Dtor. */
    virtual ~View3D_2(void);

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

    /**
     * Serialises the camera of the view
     *
     * @param serialiser Serialises the camera of the view
     */
    virtual void SerialiseCamera(vislib::Serialiser& serialiser) const;

    /**
     * Deserialises the camera of the view
     *
     * @param serialiser Deserialises the camera of the view
     */
    virtual void DeserialiseCamera(vislib::Serialiser& serialiser);

    /**
     * Renders this AbstractView3D in the currently active OpenGL context.
     *
     * @param context
     */
    virtual void Render(const mmcRenderViewContext& context);

    /**
     * Resets the view. This normally sets the camera parameters to
     * default values.
     */
    virtual void ResetView(void);

    /**
     * Resizes the AbstractView3D.
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
    /**
     * Unpacks the mouse coordinates, which are relative to the virtual
     * viewport size.
     *
     * @param x The x coordinate of the mouse position
     * @param y The y coordinate of the mouse position
     */
    virtual void unpackMouseCoordinates(float& x, float& y);

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
    /** the input modifiers corresponding to this cursor. */
    vislib::graphics::InputModifiers modkeys;
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
     * Stores the current camera settings
     *
     * @param p Must be storeCameraSettingsSlot
     *
     * @return true
     */
    bool onStoreCamera(param::ParamSlot& p);

    /**
     * Restores the camera settings
     *
     * @param p Must be restoreCameraSettingsSlot
     *
     * @return true
     */
    bool onRestoreCamera(param::ParamSlot& p);

    /**
     * Resets the view
     *
     * @param p Must be resetViewSlot
     *
     * @return true
     */
    bool onResetView(param::ParamSlot& p);

    bool onToggleButton(param::ParamSlot& p);

#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The camera */
    Camera_2 cam;

    /** The arcball manipulator for the camera */
    arcball_type arcballManipulator;

    /** The translation manipulator for the camera */
    xlate_type translateManipulator;

    /** the 2d cursor of this view */
    vislib::graphics::Cursor2D cursor2d;
#ifdef _WIN32
#    pragma warning(default : 4251)
#endif /* _WIN32 */

    /** Slot to call the renderer to render */
    CallerSlot rendererSlot;

#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The light direction vector (NOT LIGHT POSITION) */
    vislib::graphics::SceneSpaceVector3D lightDir;
#ifdef _WIN32
#    pragma warning(default : 4251)
#endif /* _WIN32 */

    /** flag whether or not the light is a camera relative light */
    bool isCamLight;

    /** Bool flag showing the bounding box */
    param::ParamSlot showBBox;

    /** Flag showing the look at point */
    param::ParamSlot showLookAt;

    /** The stored camera settings */
    param::ParamSlot cameraSettingsSlot;

    /** Triggers the storage of the camera settings */
    param::ParamSlot storeCameraSettingsSlot;

    /** Triggers the restore of the camera settings */
    param::ParamSlot restoreCameraSettingsSlot;

    /** Triggers the reset of the view */
    param::ParamSlot resetViewSlot;

    /**
     * Flag if this is the first time an image gets created. Used for
     * initial camera reset
     */
    bool firstImg;

    /**
     * Flag whether the light is relative to the camera or to the world
     * coordinate system
     */
    param::ParamSlot isCamLightSlot;

    /** Direction vector of the light */
    param::ParamSlot lightDirSlot;

    /** Diffuse light colour */
    param::ParamSlot lightColDifSlot;

    /** Ambient light colour */
    param::ParamSlot lightColAmbSlot;

    /** focus distance for stereo projection */
    param::ParamSlot stereoFocusDistSlot;

    /** eye distance for stereo projection */
    param::ParamSlot stereoEyeDistSlot;

    /** The diffuse light colour */
    float lightColDif[4];

    /** The ambient light colour */
    float lightColAmb[4];

    /** The incoming call */
    view::AbstractCallRender* overrideCall;

    /** The move step size in world coordinates */
    param::ParamSlot viewKeyMoveStepSlot;

    param::ParamSlot viewKeyRunFactorSlot;

    /** The angle rotate step in degrees */
    param::ParamSlot viewKeyAngleStepSlot;

    /** sensitivity for mouse rotation in WASD mode */
    param::ParamSlot mouseSensitivitySlot;

    /** The point around which the view will be roateted */
    param::ParamSlot viewKeyRotPointSlot;

    param::ParamSlot toggleBBoxSlot;

    param::ParamSlot hookOnChangeOnlySlot;

    /** The colour of the bounding box */
    float bboxCol[4];

    /** Parameter slot for the bounding box colour */
    param::ParamSlot bboxColSlot;

    /** Enable selecting mode of mouse (disables camera movement) */
    param::ParamSlot enableMouseSelectionSlot;

    /** Shows the view cube helper */
    param::ParamSlot showViewCubeSlot;

    /** whether to reset the view when the object bounding box changes */
    param::ParamSlot resetViewOnBBoxChangeSlot;

    /** The mouse x coordinate */
    float mouseX;

    /** The mouse y coordinate */
    float mouseY;

    /** The mouse flags */
    view::MouseFlags mouseFlags;

    /** The time control */
    view::TimeControl timeCtrl;

    /** Flag whether mouse control is to be handed over to the renderer */
    bool toggleMouseSelection;

    /** Shader program for lines */
    vislib::graphics::gl::GLSLShader lineShader;
};

} // namespace nextgen
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_View3D_2_H_INCLUDED */
