/*
 * TableSampler.h
 *
 * Copyright (C) 2020 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "datatools/table/TableDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

// Fix annoying wingdi.h defines
#undef ABSOLUTE
#undef RELATIVE

namespace megamol::datatools::table {

/*
 * Module to sample a float table.
 */
class TableSampler : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "TableSampler";
    }

    /** Return module class description */
    static const char* Description() {
        return "Randomly samples a float table.";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    TableSampler();

    /** Dtor */
    ~TableSampler() override;

protected:
    bool create() override;

    void release() override;

    bool getData(core::Call& call);

    bool getHash(core::Call& call);

    bool handleCall(core::Call& call);

    bool numberModeCallback(core::param::ParamSlot& caller);

    bool resampleCallback(core::param::ParamSlot& caller);

private:
    enum SampleNumberMode { ABSOLUTE = 0, RELATIVE = 1 };

    core::CallerSlot tableInSlot;
    core::CalleeSlot tableOutSlot;

    core::param::ParamSlot sampleNumberModeParam;
    core::param::ParamSlot sampleNumberAbsoluteParam;
    core::param::ParamSlot sampleNumberRelativeParam;
    core::param::ParamSlot resampleParam;
    // TODO control random seeding?

    // input table properties
    unsigned int tableInFrameCount;
    size_t tableInDataHash;
    size_t tableInColCount;

    // sampled table
    size_t dataHash;
    size_t rowCount;
    std::vector<TableDataCall::ColumnInfo> colInfos;
    std::vector<float> data;

    bool doResampling;
};

} // namespace megamol::datatools::table
