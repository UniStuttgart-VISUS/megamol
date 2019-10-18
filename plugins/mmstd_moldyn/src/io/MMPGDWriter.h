/*
 * MMPGDWriter.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MMPGDWRITER_H_INCLUDED
#define MEGAMOLCORE_MMPGDWRITER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/CallerSlot.h"
#include "rendering/ParticleGridDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/sys/File.h"


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace io {


/**
 * MMPGD (MegaMol Particle Grid Dump) file writer
 */
class MMPGDWriter : public core::AbstractDataWriter {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "MMPGDWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "MMPGD file writer";
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
    MMPGDWriter(void);

    /** Dtor. */
    virtual ~MMPGDWriter(void);

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
    virtual bool getCapabilities(core::DataWriterCtrlCall &call);

private:

    /**
     * Writes the data of one frame to the file
     *
     * @param file The output data file
     * @param data The data of the current frame
     *
     * @return True on success
     */
    bool writeFrame(vislib::sys::File &file, rendering::ParticleGridDataCall &data);

    /** The file name of the file to be written */
    core::param::ParamSlot filenameSlot;

    /** The slot asking for data */
    core::CallerSlot dataSlot;

};

}
} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MMPGDWRITER_H_INCLUDED */
