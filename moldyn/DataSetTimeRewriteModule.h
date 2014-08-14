/*
 * DataSetTimeRewriteModule.h
 *
 * Copyright (C) 2014 by CGV TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DATASETTIMEREWRITEMODULE_H_INCLUDED
#define MEGAMOLCORE_DATASETTIMEREWRITEMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "CallDescriptionManager.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "vislib/Cuboid.h"


namespace megamol {
namespace core {
namespace moldyn {


    /**
     * In-Between management module to change time codes of a data set
     */
    class DataSetTimeRewriteModule : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "DataSetTimeRewriteModule";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "In-Between management module to change time codes of a data set";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        DataSetTimeRewriteModule(void);

        /** Dtor. */
        virtual ~DataSetTimeRewriteModule(void);

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
         * Moves the iterator directly behind the next description of the
         * next call compatible with this module
         *
         * @param iterator The iterator to iterate
         *
         * @return The call description iterated to, or NULL if there are no
         *         more compatible calls
         */
        const CallDescription* moveToNextCompatibleCall(
            CallDescriptionManager::DescriptionIterator &iterator) const;

    private:

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getDataCallback(Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getExtentCallback(Call& caller);

        /**
         * Checks if the callee and the caller slot are connected with the
         * same call classes
         *
         * @param outCall The incoming call requesting data
         *
         * @return True if everything is fine.
         */
        bool checkConnections(Call *outCall);

        /** The slot for publishing data to the writer */
        CalleeSlot outDataSlot;

        /** The slot for requesting data from the source */
        CallerSlot inDataSlot;

        /** The number of the first frame */
        param::ParamSlot firstFrameSlot;

        /** The number of the last frame */
        param::ParamSlot lastFrameSlot;

        /** The step length between two frames */
        param::ParamSlot frameStepSlot;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DATASETTIMEREWRITEMODULE_H_INCLUDED */
