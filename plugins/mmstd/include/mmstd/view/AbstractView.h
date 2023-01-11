/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "ScriptPaths.h"
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractViewInterface.h"
#include "mmcore/view/CameraSerializer.h"
#include "mmstd/renderer/TimeControl.h"
#include "vislib/Array.h"
#include "vislib/Serialiser.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"

namespace megamol::core::view {

using megamol::frontend_resources::Key;
using megamol::frontend_resources::KeyAction;
using megamol::frontend_resources::KeyCode;
using megamol::frontend_resources::Modifier;
using megamol::frontend_resources::Modifiers;
using megamol::frontend_resources::MouseButton;
using megamol::frontend_resources::MouseButtonAction;

/**
 * Abstract base class of rendering views
 */
class AbstractView : public AbstractViewInterface {


public:
    /**
     * Interfaces class for hooking into view processes
     */
    class Hooks {
    public:
        /**
         * Empty ctor.
         */
        Hooks() {
            // intentionally empty
        }

        /**
         * Empty but virtual dtor.
         */
        virtual ~Hooks() {
            // intentionally empty
        }

        /**
         * Hook method to be called before the view is rendered.
         *
         * @param view The calling view
         */
        virtual void BeforeRender(AbstractView* view) {
            // intentionally empty
        }

        /**
         * Hook method to be called after the view is rendered.
         *
         * @param view The calling view
         */
        virtual void AfterRender(AbstractView* view) {
            // intentionally empty
        }
    };

    /** Ctor. */
    explicit AbstractView(ViewDimension dim);

    /** Dtor. */
    ~AbstractView() override;

    /**
     * Answer the default time for this view
     *
     * @param instTime the current instance time
     *
     * @return The default time
     */
    virtual float DefaultTime(double instTime) const {
        return this->_timeCtrl.Time(instTime);
    }

    /**
     * Answers whether the given parameter is relevant for this view.
     *
     * @param param The parameter to test.
     *
     * @return 'true' if 'param' is relevant, 'false' otherwise.
     */
    virtual bool IsParamRelevant(const std::shared_ptr<param::AbstractParam>& param) const;

    /**
     * Set the camera for this view externally
     *
     * @param camera A fully intialized camera to use for rendering the view
     * @param isMutable Tell the view whether it can modify, i.e. control, the camera or not
     */
    virtual void SetCamera(Camera camera, bool isMutable = true);

    /**
     * Return the current camera
     */
    virtual Camera GetCamera() const;

    /**
     * ...
     */
    virtual void CalcCameraClippingPlanes(float border);

    /**
     * Resets the view. This normally sets the camera parameters to
     * default values.
     */
    virtual void ResetView() = 0;

    /**
     * Resizes the AbstractView3D.
     *
     * @param width The new width.
     * @param height The new height.
     */
    virtual void Resize(unsigned int width, unsigned int height) = 0;

    /**
     * Registers a hook
     *
     * @param hook The hook to register
     */
    void RegisterHook(Hooks* hook) {
        if (!this->_hooks.Contains(hook)) {
            this->_hooks.Add(hook);
        }
    }

    /**
     * Unregisters a hook
     *
     * @param hook The hook to unregister
     */
    void UnregisterHook(Hooks* hook) {
        this->_hooks.RemoveAll(hook);
    }

    /**
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    virtual bool OnRenderView(Call& call);


    /**
     * Callback requesting the extents of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    virtual bool GetExtents(Call& call);

    /**
     * Restores the view
     *
     * @param p Must be resetViewSlot
     *
     * @return true
     */
    bool OnResetView(Call& call);

    bool OnKeyCallback(Call& call);

    bool OnCharCallback(Call& call);

    bool OnMouseButtonCallback(Call& call);

    bool OnMouseMoveCallback(Call& call);

    bool OnMouseScrollCallback(Call& call);

    /**
     * Answer the background colour for the view
     *
     * @return The background colour for the view
     */
    glm::vec4 BackgroundColor() const;

    /**
     * Restores the view
     *
     * @param p Must be resetViewSlot
     *
     * @return true
     */
    bool OnResetView(param::ParamSlot& p);

protected:
    void requested_lifetime_resources(frontend_resources::ResourceRequest& req) override {
        Module::requested_lifetime_resources(req);
        req.require<frontend_resources::ScriptPaths>();
    }

    /** Typedef alias */
    typedef vislib::SingleLinkedList<Hooks*>::Iterator HooksIterator;

    /**
     * Answer if hook code should be executed.
     *
     * @return 'true' if hook code should be run
     */
    inline bool doHookCode() const {
        return !this->_hooks.IsEmpty();
    }

    /**
     * Gets an iterator to the list or registered hooks.
     *
     * @return An iterator to the list of registered hooks.
     */
    inline HooksIterator getHookIterator() {
        return this->_hooks.GetIterator();
    }

    /**
     * The code triggering the pre render hook
     */
    inline void doBeforeRenderHook() {
        HooksIterator i = this->getHookIterator();
        while (i.HasNext()) {
            i.Next()->BeforeRender(this);
        }
    }

    /**
     * The code triggering the post render hook
     */
    inline void doAfterRenderHook() {
        HooksIterator i = this->getHookIterator();
        while (i.HasNext()) {
            i.Next()->AfterRender(this);
        }
    }

    /**
     * ...
     */
    void beforeRender(double time, double instanceTime);

    /**
     * ...
     */
    void afterRender();

    /**
     * Stores the current camera settings
     *
     * @param p Must be storeCameraSettingsSlot
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
     * This method determines the file path the camera file should have
     *
     * @return The file path of the camera file as string
     */
    std::string determineCameraFilePath() const;

    /**
     * Flag if this is the first time an image gets created. Used for
     * initial camera reset
     */
    bool _firstImg;

    /** Slot to call the renderer to render */
    CallerSlot _rhsRenderSlot;

    /** Slot for incoming rendering requests */
    CalleeSlot _lhsRenderSlot;

    /** The camera */
    Camera _camera;

    /** A flag that decides whether the camera is controllable by the view */
    bool _cameraIsMutable;

    /** Slot containing the settings of the currently stored camera */
    param::ParamSlot _cameraSettingsSlot;

    /** Triggers the storage of the camera settings */
    param::ParamSlot _storeCameraSettingsSlot;

    /** Triggers the restore of the camera settings */
    param::ParamSlot _restoreCameraSettingsSlot;

    /** Slot activating or deactivating the override of already present camera settings */
    param::ParamSlot _overrideCamSettingsSlot;

    /** Slot activating or deactivating the automatic save of camera parameters to disk when a camera is saved
     */
    param::ParamSlot _autoSaveCamSettingsSlot;

    /** Slot activating or deactivating the automatic load of camera parameters at program startup */
    param::ParamSlot _autoLoadCamSettingsSlot;

    /** Triggers the reset of the view */
    param::ParamSlot _resetViewSlot;

    /** whether to reset the view when the object bounding box changes */
    param::ParamSlot _resetViewOnBBoxChangeSlot;

    /** Flag showing the look at point */
    param::ParamSlot _showLookAt;

    /** Shows the view cube helper */
    param::ParamSlot _showViewCubeParam;

    /** Array that holds the saved camera states */
    std::array<std::pair<Camera, bool>, 11> _savedCameras;

    /** The object responsible for camera serialization */
    CameraSerializer _cameraSerializer;

    /** The time control */
    view::TimeControl _timeCtrl;

    /**  */
    std::chrono::time_point<std::chrono::high_resolution_clock> _lastFrameTime;

    std::chrono::microseconds _lastFrameDuration;

    /** The background colour for the view */
    mutable param::ParamSlot _backgroundColSlot;

private:
    /** List of registered hooks */
    vislib::SingleLinkedList<Hooks*> _hooks;

    /** The background colour for the view */
    mutable glm::vec4 _backgroundCol;
};

} // namespace megamol::core::view
