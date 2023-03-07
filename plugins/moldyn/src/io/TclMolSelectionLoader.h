/*
 * TclMolSelectionLoader.h
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "geometry_calls/ParticleRelistCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include <vector>

namespace megamol::moldyn::io {

/**
 * Data loader for tcl files containing mol commands for selection serials:
 *
 * mol selection serial {}
 * mol addrep *
 *
 */
class TclMolSelectionLoader : public core::Module {
public:
    static const char* ClassName() {
        return "TclMolSelectionLoader";
    }
    static const char* Description() {
        return "Data loader for tcl files containing mol commands for selection serials";
    }
    static bool IsAvailable() {
        return true;
    }

    TclMolSelectionLoader();
    ~TclMolSelectionLoader() override;

protected:
    bool create() override;
    void release() override;

private:
    bool getDataCallback(core::Call& caller);

    void clear();
    void load();

    core::CalleeSlot getDataSlot;

    core::param::ParamSlot filenameSlot;

    size_t hash;
    geocalls::ParticleRelistCall::ListIDType cnt;
    std::vector<geocalls::ParticleRelistCall::ListIDType> data;
};

} // namespace megamol::moldyn::io
