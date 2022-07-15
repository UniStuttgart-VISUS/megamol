/*
 * ParticleRelistCall.h
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include <cstdint>

namespace megamol::geocalls {

/**
 * Class for binding site calls and data interfaces.
 *
 * @remarks: class derives from "AbstractGetData3DCall" and uses that
 *           interface to be compatible with meta-modules, like
 *           "DataFileSequence". "GetExtent" does not transport meaningful
 *           data!
 */
class ParticleRelistCall : public core::AbstractGetData3DCall {
public:
    typedef uint16_t ListIDType;

    static const char* ClassName(void) {
        return "ParticleRelistCall";
    }
    static const char* Description(void) {
        return "Call to get relist information for MultiParticleDataCall reorganization";
    }
    static unsigned int FunctionCount(void) {
        return 2;
    }
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetData";
        case 1:
            return "GetExtent";
        }
        return nullptr;
    }

    ParticleRelistCall(void);
    virtual ~ParticleRelistCall(void);

    inline ListIDType TargetListCount(void) const {
        return tarListCount;
    }
    inline uint64_t SourceParticleCount(void) const {
        return srcPartCount;
    }
    inline const ListIDType* SourceParticleTargetLists(void) const {
        return srcParticleTarLists;
    }

    inline void Set(ListIDType tlc, uint64_t spc, const ListIDType* sptl) {
        tarListCount = tlc;
        srcPartCount = spc;
        srcParticleTarLists = sptl;
    }

private:
    /** number of target lists to create */
    unsigned int tarListCount;
    /** number of particles in the source list */
    uint64_t srcPartCount;
    /** For each particle in the source list, the id of the target list to be copied to */
    const ListIDType* srcParticleTarLists;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<ParticleRelistCall> ParticleRelistCallDescription;

} /* end namespace megamol::geocalls */
