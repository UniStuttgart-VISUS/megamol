/*
 * SIFFWriter.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SIFFWRITER_H_INCLUDED
#define MEGAMOLCORE_SIFFWRITER_H_INCLUDED
#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AbstractDataWriter.h"
#include "vislib/sys/File.h"


namespace megamol::moldyn::io {

/**
 * SIFF writer module
 */
class SIFFWriter : public core::AbstractDataWriter {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "SIFFWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Writing SIFF";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    SIFFWriter();

    /** Dtor. */
    ~SIFFWriter() override;

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
     * The main function
     *
     * @return True on success
     */
    bool run() override;

    /**
     * Function querying the writers capabilities
     *
     * @param call The call to receive the capabilities
     *
     * @return True on success
     */
    bool getCapabilities(core::DataWriterCtrlCall& call) override;

private:
    /** The file name of the file to be written */
    core::param::ParamSlot filenameSlot;

    /** The slot asking for data */
    core::param::ParamSlot asciiSlot;

    /** The slot asking for data */
    core::param::ParamSlot versionSlot;

    /** The slot asking for data */
    core::CallerSlot dataSlot;
};

} // namespace megamol::moldyn::io

#endif /* MEGAMOLCORE_SIFFWRITER_H_INCLUDED */
