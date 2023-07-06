/**
 * MegaMol
 * Copyright (c) 2016, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include "datatools/table/TableDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AbstractDataWriter.h"

namespace megamol::datatools::table {

/**
 * MMFTDataWriter (MegaMol Particle List Dump) file writer
 */
class MMFTDataWriter : public core::AbstractDataWriter {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "MMFTDataWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Binary float table data file writer";
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
    MMFTDataWriter();

    /** Dtor. */
    ~MMFTDataWriter() override;

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
    core::CallerSlot dataSlot;
};

} // namespace megamol::datatools::table
