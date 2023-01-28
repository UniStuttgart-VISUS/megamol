/*
 * DatRawWriter.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DATRAWWRITER_H_INCLUDED
#define MEGAMOLCORE_DATRAWWRITER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AbstractDataWriter.h"
#include "mmstd/data/DataWriterCtrlCall.h"
#include <fstream>

namespace megamol::volume {

/*
 * Writer for volume data
 */
class DatRawWriter : public megamol::core::AbstractDataWriter {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "DatRawWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Dat-Raw file writer";
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
    DatRawWriter();

    /** Dtor. */
    ~DatRawWriter() override;

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
    /**
     * Writes the data of one frame to the file
     *
     * @param datpath The file path to the dat file
     * @param rawpath The file path to the raw file
     * @param data The data of the current frame
     *
     * @return True on success
     */
    bool writeFrame(std::string datpath, std::string rawpath, geocalls::VolumetricDataCall& data);

    /** The file name of the file to be written */
    core::param::ParamSlot filenameSlot;

    /** The frame ID of the frame to be written */
    core::param::ParamSlot frameIDSlot;

    /** The slot asking for data */
    core::CallerSlot dataSlot;
};

} // namespace megamol::volume

#endif /* MEGAMOLCORE_DATRAWWRITER_H_INCLUDED */
