/*
 * ParticleRelistCall.h
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_MOLDYN_PARTICLERELISTCALL_H_INCLUDED
#define MEGAMOLCORE_MOLDYN_PARTICLERELISTCALL_H_INCLUDED
#pragma once

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include <cstdint>

namespace megamol {
namespace core {
namespace moldyn {

    /**
     * Class for binding site calls and data interfaces.
     */

    class MEGAMOLCORE_API ParticleRelistCall : public AbstractGetDataCall {
    public:
        typedef uint16_t ListIDType;

        static const char *ClassName(void) { return "ParticleRelistCall"; }
        static const char *Description(void) { return "Call to get relist information for MultiParticleDataCall reorganization"; }
        static unsigned int FunctionCount(void) { return 1; }
        static const char* FunctionName(unsigned int) { return "get"; }

        ParticleRelistCall(void);
        virtual ~ParticleRelistCall(void);

        inline ListIDType TargetListCount(void) const {
            return tarListCount;
        }
        inline uint64_t SourceParticleCount(void) const {
            return srcPartCount;
        }
        inline const ListIDType * SourceParticleTargetLists(void) const {
            return srcParticleTarLists;
        }

        inline void Set(ListIDType tlc, uint64_t spc, const ListIDType *sptl) {
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
        const ListIDType *srcParticleTarLists;
    };

    /** Description class typedef */
    typedef megamol::core::factories::CallAutoDescription<ParticleRelistCall> ParticleRelistCallDescription;

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MOLDYN_PARTICLERELISTCALL_H_INCLUDED */
