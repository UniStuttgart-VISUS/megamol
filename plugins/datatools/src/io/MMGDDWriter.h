/*
 * MMGDDWriter.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_IO_MMGDDWRITER_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_MMGDDWRITER_H_INCLUDED
#pragma once

#include "datatools/GraphDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AbstractDataWriter.h"
#include "vislib/sys/File.h"


namespace megamol {
namespace datatools {
namespace io {


/**
 * MegaMol GraphData Dump file writer
 */
class MMGDDWriter : public core::AbstractDataWriter {
public:
    static const char* ClassName(void) {
        return "MMGDDWriter";
    }
    static const char* Description(void) {
        return "MegaMol GraphData Dump file writer";
    }
    static bool IsAvailable(void) {
        return true;
    }

    MMGDDWriter(void);
    ~MMGDDWriter(void) override;

protected:
    bool create(void) override;
    void release(void) override;

    bool run(void) override;

    bool getCapabilities(core::DataWriterCtrlCall& call) override;

private:
    core::param::ParamSlot filenameSlot;
    core::CallerSlot dataSlot;
};

} /* end namespace io */
} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MMPLDWRITER_H_INCLUDED */
