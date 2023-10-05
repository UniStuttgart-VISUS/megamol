/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "ModuleGraphSubscription.h"

#include <glowl/FramebufferObject.hpp>

#include "mmcore/MegaMolGraph.h"
#include "mmcore/Module.h"
#include "mmcore/job/AbstractJob.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/view/AbstractView.h"
#include "RuntimeInfo.h"

namespace megamol::mmstd_gl::special {

/**
 * Class implementing the screen shooter job module
 */
class ScreenShooter : public core::job::AbstractJob, public core::Module, public core::view::AbstractView::Hooks {
public:
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        Module::requested_lifetime_resources(req);
        req.require<core::MegaMolGraph>();
        req.require<frontend_resources::RuntimeInfo>();
    }

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ScreenShooter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "A simple job module used to create large off-screen renderings";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable();

    /**
     * Ctor
     *
     * @param reducedParameters True: only show necessary parameters, false: all
     */
    explicit ScreenShooter(bool reducedParameters = false);

    /**
     * Dtor
     */
    ~ScreenShooter() override;

    /**
     * Answers whether or not this job is still running.
     *
     * @return 'true' if this job is still running, 'false' if it has
     *         finished.
     */
    bool IsRunning() const override;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    bool Start() override;

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    bool Terminate() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * Hook method to be called before the view is rendered.
     *
     * @param view The calling view
     */
    void BeforeRender(core::view::AbstractView* view) override;

    /*
     * Create the screenshot.
     *
     * @param filename Filename of the output screenshot
     */
    void createScreenshot(const std::string& filename);

private:
    /**
     * Starts the image making triggered by clicking on the trigger button.
     *
     * @param slot Must be the triggerButtonSlot
     */
    bool triggerButtonClicked(core::param::ParamSlot& slot);

    core::param::ParamSlot* findTimeParam(core::view::AbstractView* view);

    /** The name of the view instance to be shot */
    core::param::ParamSlot viewNameSlot;

    /** The width in pixel of the resulting image */
    core::param::ParamSlot imgWidthSlot;

    /** The height in pixel of the resulting image */
    core::param::ParamSlot imgHeightSlot;

    /** The width of a rendering tile in pixel */
    core::param::ParamSlot tileWidthSlot;

    /** The height of a rendering tile in pixel */
    core::param::ParamSlot tileHeightSlot;

    /** The file name to store the resulting image under */
    core::param::ParamSlot imageFilenameSlot;

    /** Enum controlling the background to be used */
    core::param::ParamSlot backgroundSlot;

    /** The trigger button */
    core::param::ParamSlot triggerButtonSlot;

    /** Bool whether or not to close the application after the screen shot was taken */
    core::param::ParamSlot closeAfterShotSlot;

    core::param::ParamSlot animFromSlot;
    core::param::ParamSlot animToSlot;
    core::param::ParamSlot animStepSlot;
    core::param::ParamSlot animAddTime2FrameSlot;
    core::param::ParamSlot makeAnimSlot;
    core::param::ParamSlot animTimeParamNameSlot;
    core::param::ParamSlot disableCompressionSlot;
    float animLastFrameTime;
    int outputCounter;

    /** A simple running flag */
    bool running;

    std::shared_ptr<glowl::FramebufferObject> currentFbo;

    frontend_resources::RuntimeInfo const* ri_ = nullptr;
};

} // namespace megamol::mmstd_gl::special
