/*
 * TclMolSelectionLoader.h
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_STDMOLDYN_TCLMOLSELECTIONLOADER_H_INCLUDED
#define MEGAMOL_STDMOLDYN_TCLMOLSELECTIONLOADER_H_INCLUDED
#pragma once

#include "geometry_calls/ParticleRelistCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include <vector>

namespace megamol {
namespace moldyn {
namespace io {

/**
 * Data loader for tcl files containing mol commands for selection serials:
 *
 * mol selection serial {}
 * mol addrep *
 *
 */
class TclMolSelectionLoader : public core::Module {
public:
    static const char* ClassName(void) {
        return "TclMolSelectionLoader";
    }
    static const char* Description(void) {
        return "Data loader for tcl files containing mol commands for selection serials";
    }
    static bool IsAvailable(void) {
        return true;
    }

    TclMolSelectionLoader();
    ~TclMolSelectionLoader() override;

protected:
    bool create(void) override;
    void release(void) override;

private:
    bool getDataCallback(core::Call& caller);

    void clear(void);
    void load(void);

    core::CalleeSlot getDataSlot;

    core::param::ParamSlot filenameSlot;

    size_t hash;
    geocalls::ParticleRelistCall::ListIDType cnt;
    std::vector<geocalls::ParticleRelistCall::ListIDType> data;
};

} /* end namespace io */
} /* end namespace moldyn */
} /* end namespace megamol */

#endif /* MEGAMOL_STDMOLDYN_TCLMOLSELECTIONLOADER_H_INCLUDED */
