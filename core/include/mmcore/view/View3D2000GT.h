/*
 * View3D2000GT.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIEW3D2000GT_H_INCLUDED
#define MEGAMOLCORE_VIEW3D2000GT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <memory>
#include "AbstractCamParamSync.h"
#include "mmcore/BoundingBoxes.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractCallRender.h"
#include "mmcore/view/AbstractView3D.h"
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
namespace view {

class MEGAMOLCORE_API View3D2000GT : public AbstractView3D, public AbstractCamParamSync {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "View3D2000GT"; }

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
    View3D2000GT(void);

    /** Dtor. */
    virtual ~View3D2000GT(void);

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
     * Sets the button state of a button of the 2d cursor. See
     * 'vislib::graphics::Cursor2D' for additional information.
     *
     * @param button The button.
     * @param down Flag whether the button is pressed, or not.
     */
    virtual void SetCursor2DButtonState(unsigned int btn, bool down);

    /**
     * Sets the position of the 2d cursor. See 'vislib::graphics::Cursor2D'
     * for additional information.
     *
     * @param x The x coordinate
     * @param y The y coordinate
     */
    virtual void SetCursor2DPosition(float x, float y);

    /**
     * Sets the state of an input modifier.
     *
     * @param mod The input modifier to be set.
     * @param down The new state of the input modifier.
     */
    virtual void SetInputModifier(mmcInputModifier mod, bool down);

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

    /** The normal camera parameters */
    vislib::SmartPtr<vislib::graphics::CameraParameters> camParams;
#ifdef _WIN32
#    pragma warning(default : 4251)
#endif /* _WIN32 */

    /** The complete scene bounding box */
    BoundingBoxes bboxs;

#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** keeping track of changes in the camera between frames */
    vislib::SmartPtr<vislib::graphics::CameraParamsStore> lastFrameParams = new vislib::graphics::CameraParamsStore();
#ifdef _WIN32
#    pragma warning(default : 4251)
#endif /* _WIN32 */

    bool frameIsNew = false;

    /**
     * internal utility class storing frozen values
     */
    class FrozenValues {
    public:
        /**
         * Ctor
         */
        FrozenValues(void) {
            this->camParams = new vislib::graphics::CameraParamsStore();
            this->time = 0.0f;
            this->freezeCounter = 1;
        }

        /** The camera parameters frozen (does not work at all!) */
        vislib::SmartPtr<vislib::graphics::CameraParameters> camParams;

        /** The frame time frozen */
        float time;

        /** The freezeCounter */
        unsigned int freezeCounter;
    };

    /**
     * Renders the vertices of the bounding box
     */
    inline void renderBBox(void);

    /**
     * Renders the back side of the bounding box
     */
    inline void renderBBoxBackside(void);

    /**
     * Renders the front side of the bounding box
     */
    inline void renderBBoxFrontside(void);

    /**
     * Renders the cross for the look-at point
     */
    void renderLookAt(void);

    /**
     * Renders the soft cursor
     */
    void renderSoftCursor(void);

    bool mouseSensitivityChanged(param::ParamSlot& p);

    /**
     * Handles a request for the camera parameters used by the view.
     *
     * @param c The call being executed.
     *
     * @return true in case of success, false otherwise.
     */
    virtual bool OnGetCamParams(CallCamParamSync& c);

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

    /**
     * Event handler for view keys
     *
     * @param p The parameter slot of the view key hit
     *
     * @return true
     */
    bool viewKeyPressed(param::ParamSlot& p);

    bool onToggleButton(param::ParamSlot& p);

    /**
     * Renders the view cube
     */
    void renderViewCube(void);

    ///**
    // * Renders a single egde of the view cube
    // *
    // * @param x1
    // * @param y1
    // * @param z1
    // * @param x2
    // * @param y2
    // * @param z2
    // */
    // inline void renderViewCubeEdge(int x1, int y1, int z1, int x2, int y2, int z2);

#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The scene camera */
    vislib::graphics::gl::CameraOpenGL cam;

    /** The camera parameter overrides */
    vislib::SmartPtr<vislib::graphics::CameraParameters> camOverrides;

    /** the 2d cursor of this view */
    vislib::graphics::Cursor2D cursor2d;

    /** camera look at rotator */
    vislib::graphics::CameraRotate2DLookAt rotator1;

    /** camera rotator */
    vislib::graphics::CameraRotate2D rotator2;

    /** camera move zoom */
    vislib::graphics::CameraZoom2DMove zoomer1;

    /** camera angle zoom */
    vislib::graphics::CameraZoom2DAngle zoomer2;

    /** camera mover */
    vislib::graphics::CameraMove2D mover;

    /** camera look-at distance changer */
    vislib::graphics::CameraLookAtDist lookAtDist;
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

#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The frozen values */
    FrozenValues* frozenValues;
#ifdef _WIN32
#    pragma warning(default : 4251)
#endif /* _WIN32 */

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
    AbstractCallRender* overrideCall;

    /** The move step size in world coordinates */
    param::ParamSlot viewKeyMoveStepSlot;

    param::ParamSlot viewKeyRunFactorSlot;

    /** The angle rotate step in degrees */
    param::ParamSlot viewKeyAngleStepSlot;

    /** sensitivity for mouse rotation in WASD mode */
    param::ParamSlot mouseSensitivitySlot;

    /** The point around which the view will be roateted */
    param::ParamSlot viewKeyRotPointSlot;

    /** Rotates the view to the left (around the up-axis) */
    param::ParamSlot viewKeyRotLeftSlot;

    /** Rotates the view to the right (around the up-axis) */
    param::ParamSlot viewKeyRotRightSlot;

    /** Rotates the view to the top (around the right-axis) */
    param::ParamSlot viewKeyRotUpSlot;

    /** Rotates the view to the bottom (around the right-axis) */
    param::ParamSlot viewKeyRotDownSlot;

    /** Rotates the view counter-clockwise (around the view-axis) */
    param::ParamSlot viewKeyRollLeftSlot;

    /** Rotates the view clockwise (around the view-axis) */
    param::ParamSlot viewKeyRollRightSlot;

    /** Zooms in */
    param::ParamSlot viewKeyZoomInSlot;

    /** Zooms out */
    param::ParamSlot viewKeyZoomOutSlot;

    /** Moves to the left */
    param::ParamSlot viewKeyMoveLeftSlot;

    /** Moves to the right */
    param::ParamSlot viewKeyMoveRightSlot;

    /** Moves to the top */
    param::ParamSlot viewKeyMoveUpSlot;

    /** Moves to the bottom */
    param::ParamSlot viewKeyMoveDownSlot;

    param::ParamSlot toggleBBoxSlot;

    param::ParamSlot toggleSoftCursorSlot;

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
    MouseFlags mouseFlags;

    /** The time control */
    TimeControl timeCtrl;

    /** Flag whether mouse control is to be handed over to the renderer */
    bool toggleMouseSelection;

    /** Shader program for lines */
    vislib::graphics::gl::GLSLShader lineShader;

private:
    /**********************************************************************
     * variables
     **********************************************************************/

    enum corner {
        TOP_LEFT = 0,
        TOP_RIGHT = 1,
        BOTTOM_LEFT = 2,
        BOTTOM_RIGHT = 3,
    };

#ifdef _WIN32
#    pragma warning(disable : 4251)
#endif /* _WIN32 */
    vislib::graphics::gl::OpenGLTexture2D textureTopLeft;
    vislib::graphics::gl::OpenGLTexture2D textureTopRight;
    vislib::graphics::gl::OpenGLTexture2D textureBottomLeft;
    vislib::graphics::gl::OpenGLTexture2D textureBottomRight;

    vislib::math::Vector<float, 2> sizeTopLeft;
    vislib::math::Vector<float, 2> sizeTopRight;
    vislib::math::Vector<float, 2> sizeBottomLeft;
    vislib::math::Vector<float, 2> sizeBottomRight;
#ifdef _WIN32
#    pragma warning(default : 4251)
#endif /* _WIN32 */

    float lastScaleAll;
    bool firstParamChange;

    /**********************************************************************
     * functions
     **********************************************************************/

    /*   */
    SIZE_T loadFile(vislib::StringA name, void** outData);

    /**  */
    bool loadTexture(View3D2000GT::corner cor, vislib::StringA filename);

    /**  */
    bool renderWatermark(View3D2000GT::corner cor, float vpH, float vpW);

    /**********************************************************************
     * parameters
     **********************************************************************/

    /**  */
    core::param::ParamSlot paramImgTopLeft;
    core::param::ParamSlot paramImgTopRight;
    core::param::ParamSlot paramImgBottomLeft;
    core::param::ParamSlot paramImgBottomRight;

    /**  */
    core::param::ParamSlot paramScaleAll;
    core::param::ParamSlot paramScaleTopLeft;
    core::param::ParamSlot paramScaleTopRight;
    core::param::ParamSlot paramScaleBottomLeft;
    core::param::ParamSlot paramScaleBottomRight;

    /**  */
    core::param::ParamSlot paramAlpha;
};

} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEW3D2000GT_H_INCLUDED */
