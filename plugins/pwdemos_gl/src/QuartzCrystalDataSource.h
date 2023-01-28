/*
 * QuartzCrystalDataSource.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "QuartzCrystalDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"


namespace megamol::demos_gl {

/**
 * Module loading a quartz crystal definition file
 */
class CrystalDataSource : public megamol::core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "QuartzCrystalDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module loading a quartz crystal definition file";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    CrystalDataSource();

    /** Dtor */
    ~CrystalDataSource() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Call callback to get the data
     *
     * @param c The calling call
     *
     * @return True on success
     */
    bool getData(core::Call& c);

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /** The file name slot */
    core::param::ParamSlot filenameSlot;

    /** The data callee slot */
    core::CalleeSlot dataOutSlot;

    /**
     * Loads a BDMD file
     *
     * @param filename The path to the file to load
     */
    void loadFile(const vislib::TString& filename);

    /** The data hash */
    SIZE_T datahash;

    /** The crystals */
    vislib::Array<CrystalDataCall::Crystal> crystals;
};

} // namespace megamol::demos_gl
