/*
 * MoleculeBallifier.h
 *
 * Copyright (C) 2012 by TU Dresden
 * All rights reserved.
 */

#ifndef MMPROTEINCUDAPLUGIN_MOLECULEBALLIFIER_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_MOLECULEBALLIFIER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "vislib/RawStorage.h"

namespace megamol {
namespace protein_cuda {

    class MoleculeBallifier : public megamol::core::Module {
    public:
        static const char *ClassName(void) {
            return "MoleculeBallifier";
        }
        static const char *Description(void) {
            return "MoleculeBallifier";
        }
        static bool IsAvailable(void) {
            return true;
        }
        MoleculeBallifier(void);
        virtual ~MoleculeBallifier(void);
    protected:
        virtual bool create(void);
        virtual void release(void);
    private:
        bool getData(core::Call& c);
        bool getExt(core::Call& c);

        core::CalleeSlot outDataSlot;
        core::CallerSlot inDataSlot;
        SIZE_T inHash, outHash;
        vislib::RawStorage data;
        float colMin, colMax;
        int frameOld;
    };

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MMPROTEINCUDAPLUGIN_MOLECULEBALLIFIER_H_INCLUDED */
