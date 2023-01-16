/*
 * WavefrontObjWriter.h
 *
 * Copyright (C) 2016 by Karsten Schatz
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_WAVEFRONTOBJWRITER_H_INCLUDED
#define MMTRISOUPPLG_WAVEFRONTOBJWRITER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "geometry_calls/LinesDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AbstractDataWriter.h"
#include "vislib/sys/FastFile.h"

namespace megamol {
namespace trisoup {

/**
 * Wavefront .obj file writer that discards the color information.
 * The .obj file writer currently only supports LinesDataCalls.
 */
class WavefrontObjWriter : public core::AbstractDataWriter {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "WavefrontObjWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Wavefront .obj file writer";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    WavefrontObjWriter(void);

    /** Dtor. */
    virtual ~WavefrontObjWriter(void);

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
     * The main function
     *
     * @return True on success
     */
    virtual bool run(void);

    /**
     * Function querying the writers capabilities
     *
     * @param call The call to receive the capabilities
     *
     * @return True on success
     */
    virtual bool getCapabilities(core::DataWriterCtrlCall& call);

private:
    /**
     * Function writing the content of a LinesDataCall to disk.
     *
     * @param ldc Pointer to the LinesDataCall
     */
    bool writeLines(megamol::geocalls::LinesDataCall* ldc);

    /** The file name of the file to be written */
    core::param::ParamSlot filenameSlot;

    /** The frame ID of the frame to be written */
    core::param::ParamSlot frameIDSlot;

    /** The slot asking for data. */
    core::CallerSlot dataSlot;
};

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MMTRISOUPPLG_WAVEFRONTOBJWRITER_H_INCLUDED */
