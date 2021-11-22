/*
 * ScreenShooter.h
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SCREENSHOOTER_H_INCLUDED
#define MEGAMOLCORE_SCREENSHOOTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/ViewInstance.h"
#include "mmcore/job/AbstractJob.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractView.h"

#include "glowl/FramebufferObject.hpp"


namespace megamol {
namespace core_gl {
namespace view {
namespace special {


/**
 * Class implementing the screen shooter job module
 */
class ScreenShooter : public core::job::AbstractJob, public core::Module, public core::view::AbstractView::Hooks {
public:
    std::vector<std::string> requested_lifetime_resources() {
        auto lifetime_resources = Module::requested_lifetime_resources();
        lifetime_resources.push_back("MegaMolGraph");
        return lifetime_resources;
    }

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ScreenShooter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "A simple job module used to create large off-screen renderings";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void);

    /**
     * Disallow usage in quickstarts
     *
     * @return false
     */
    static bool SupportQuickstart(void) {
        return false;
    }

    /**
     * Ctor
     *
     * @param reducedParameters True: only show necessary parameters, false: all
     */
    explicit ScreenShooter(bool reducedParameters = false);

    /**
     * Dtor
     */
    virtual ~ScreenShooter();

    /**
     * Answers whether or not this job is still running.
     *
     * @return 'true' if this job is still running, 'false' if it has
     *         finished.
     */
    virtual bool IsRunning(void) const;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    virtual bool Start(void);

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    virtual bool Terminate(void);

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
    virtual void release(void);

    /**
     * Hook method to be called before the view is rendered.
     *
     * @param view The calling view
     */
    virtual void BeforeRender(core::view::AbstractView* view);

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
};


} /* end namespace special */
} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SCREENSHOOTER_H_INCLUDED */
