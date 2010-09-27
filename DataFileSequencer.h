/*
 * DataFileSequencer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DATAFILESEQUENCER_H_INCLUDED
#define MEGAMOLCORE_DATAFILESEQUENCER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "param/ParamSlot.h"
#include "CallerSlot.h"


namespace megamol {
namespace core {

    /**
     * Class for manually stepping through a series of data files
     */
    class DataFileSequencer : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "DataFileSequencer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Controller for a sequence for data files";
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

        /*
         * DataFileSequencer
         */
        DataFileSequencer(void);

        /*
         * ~DataFileSequencer
         */
        virtual ~DataFileSequencer(void);

    private:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Searches for the filename slot in the module graph
         *
         * @return The found filename slot or NULL if the slot was not found.
         */
        param::ParamSlot * findFilenameSlot(void);

        /**
         * Gets the filename value
         *
         * @param slot The parameter slot
         * @param outName The string receiving the value
         *
         * @return True on success
         */
        bool GetFilename(param::ParamSlot& slot, vislib::TString& outName) const;

        /**
         * Transforms the filename into a format string
         *
         * @param inoutName The filename to be transformed
         * @param outValue The number value of the current file
         *
         * @return True on success
         */
        bool GetFormat(vislib::TString& inoutName, int& outValue) const;

        /**
         * Switches to the next file
         *
         * @param param Must be 'nextFileSlot'
         *
         * @return true
         */
        bool onNextFile(param::ParamSlot& param);

        /**
         * Switches to the previous file
         *
         * @param param Must be 'prevFileSlot'
         *
         * @return true
         */
        bool onPrevFile(param::ParamSlot& param);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * Sets the filename value
         *
         * @param slot The parameter slot
         * @param name The new value
         *
         * @return True on success
         */
        bool SetFilename(param::ParamSlot& slot, const vislib::TString& name) const;

        /**
         * Connection to the view manipulated to make this module part of the
         * same module graph
         */
        CallerSlot conSlot;

        /** String parameter identifying the filename slot to manipulate */
        param::ParamSlot filenameSlotNameSlot;

        /** Button parameter to switch to the next file */
        param::ParamSlot nextFileSlot;

        /** Button parameter to switch to the previous file */
        param::ParamSlot prevFileSlot;

    };

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DATAFILESEQUENCER_H_INCLUDED */
