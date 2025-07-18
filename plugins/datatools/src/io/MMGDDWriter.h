/*
 * MMGDDWriter.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "datatools/GraphDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AbstractDataWriter.h"
#include "vislib/sys/File.h"


namespace megamol::datatools::io {


/**
 * MegaMol GraphData Dump file writer
 */
class MMGDDWriter : public core::AbstractDataWriter {
public:
    static const char* ClassName() {
        return "MMGDDWriter";
    }
    static const char* Description() {
        return "MegaMol GraphData Dump file writer";
    }
    static bool IsAvailable() {
        return true;
    }

    MMGDDWriter();
    ~MMGDDWriter() override;

protected:
    bool create() override;
    void release() override;

    bool run() override;

private:
    core::param::ParamSlot filenameSlot;
    core::CallerSlot dataSlot;
};

} // namespace megamol::datatools::io
