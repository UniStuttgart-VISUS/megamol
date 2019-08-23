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

#include "mmcore/job/AbstractJob.h"
#include "mmcore/Module.h"
#include "mmcore/ViewInstance.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractView.h"


namespace megamol {
namespace core {
namespace view {
namespace special {


    /**
     * Class implementing the screen shooter job module
     */
    class ScreenShooter : public job::AbstractJob, public Module,
        public view::AbstractView::Hooks {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ScreenShooter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
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
        virtual void BeforeRender(view::AbstractView *view);

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
        bool triggerButtonClicked(param::ParamSlot& slot);

        param::ParamSlot* findTimeParam(view::AbstractView* view);

        /** The name of the view instance to be shot */
        param::ParamSlot viewNameSlot;

        /** The width in pixel of the resulting image */
        param::ParamSlot imgWidthSlot;

        /** The height in pixel of the resulting image */
        param::ParamSlot imgHeightSlot;

        /** The width of a rendering tile in pixel */
        param::ParamSlot tileWidthSlot;

        /** The height of a rendering tile in pixel */
        param::ParamSlot tileHeightSlot;

        /** The file name to store the resulting image under */
        param::ParamSlot imageFilenameSlot;

        /** Enum controlling the background to be used */
        param::ParamSlot backgroundSlot;

        /** The trigger button */
        param::ParamSlot triggerButtonSlot;

        /** Bool whether or not to close the application after the screen shot was taken */
        param::ParamSlot closeAfterShotSlot;

        param::ParamSlot animFromSlot;
        param::ParamSlot animToSlot;
        param::ParamSlot animStepSlot;
        param::ParamSlot animAddTime2FrameSlot;
        param::ParamSlot makeAnimSlot;
        param::ParamSlot animTimeParamNameSlot;
        param::ParamSlot disableCompressionSlot;
        float animLastFrameTime;
        int outputCounter;

        /** A simple running flag */
        bool running;

    };


} /* end namespace special */
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SCREENSHOOTER_H_INCLUDED */
