/*
 * DumpIColorHistogramModule.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DUMPICOLORHISTOGRAMMODULE_H_INCLUDED
#define MEGAMOLCORE_DUMPICOLORHISTOGRAMMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace stdplugin {
namespace datatools {

    class DumpIColorHistogramModule : public megamol::core::Module {
    public:
        static const char *ClassName(void) {
            return "DumpIColorHistogramModule";
        }
        static const char *Description(void) {
            return "Dump (DEBUG! DO NOT USE)";
        }
        static bool IsAvailable(void) {
            return true;
        }
        DumpIColorHistogramModule(void);
        virtual ~DumpIColorHistogramModule(void);
        virtual bool create(void);
        virtual void release(void);
    private:
        bool dump(::megamol::core::param::ParamSlot& param);
        megamol::core::CallerSlot inDataSlot;
        megamol::core::param::ParamSlot dumpBtnSlot;
        megamol::core::param::ParamSlot timeSlot;
    };


} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DUMPICOLORHISTOGRAMMODULE_H_INCLUDED */
