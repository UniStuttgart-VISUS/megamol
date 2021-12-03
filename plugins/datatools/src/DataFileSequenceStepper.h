/*
 * DataFileSequenceStepper.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace datatools {

/**
 * Class for manually stepping through a series of data files
 *
 * TODO: This should be replaces as soon as the parameter slots are
 *       replaced and support connections.
 */
class DataFileSequenceStepper : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "DataFileSequenceStepper";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Small utility module adding user controls to step though a sequence of data files.";
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
     * DataFileSequenceStepper
     */
    DataFileSequenceStepper(void);

    /*
     * ~DataFileSequenceStepper
     */
    virtual ~DataFileSequenceStepper(void);

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
    core::param::ParamSlot* findFilenameSlot(void);

    /**
     * Gets the filename value
     *
     * @param slot The parameter slot
     * @param outName The string receiving the value
     *
     * @return True on success
     */
    bool GetFilename(core::param::ParamSlot& slot, vislib::TString& outName) const;

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
    bool onNextFile(core::param::ParamSlot& param);

    /**
     * Switches to the previous file
     *
     * @param param Must be 'prevFileSlot'
     *
     * @return true
     */
    bool onPrevFile(core::param::ParamSlot& param);

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
    bool SetFilename(core::param::ParamSlot& slot, const vislib::TString& name) const;

    /**
     * Connection to the view manipulated to make this module part of the
     * same module graph
     */
    core::CallerSlot conSlot;

    /** String parameter identifying the filename slot to manipulate */
    core::param::ParamSlot filenameSlotNameSlot;

    /** Button parameter to switch to the next file */
    core::param::ParamSlot nextFileSlot;

    /** Button parameter to switch to the previous file */
    core::param::ParamSlot prevFileSlot;
};

} /* end namespace datatools */
} /* end namespace megamol */
