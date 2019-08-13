/*
 * MinMax.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_ASTRO_MINMAX_H_INCLUDED
#define MEGAMOL_ASTRO_MINMAX_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <array>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/misc/VolumetricDataCall.h"


namespace megamol {
namespace astro {

    /// <summary>
    /// Gets min/max values on a <see cref="VolumetricDataCall" />.
    /// </summary>
    class MinMax : public core::Module {

    public:

        static inline const char *ClassName(void) {
            return "MinMax";
        }

        static inline const char *Description(void) {
            return "Gets min/max values over all frames of a VolumetricDataCall";
        }

        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        MinMax(void);

        /** Dtor. */
        virtual ~MinMax(void);

    protected:

        virtual bool create(void);

        virtual void release(void);

        bool onGetData(core::Call& call);

        bool onGetExtents(core::Call& call);

        bool onGetMetadata(core::Call& call);

        bool onStartAsync(core::Call& call);

        bool onStopAsync(core::Call& call);

        bool onTryGetData(core::Call& call);

        bool pipeVolumetricDataCall(core::Call& call, unsigned int funcIdx);

    private:

        core::CallerSlot slotVolumetricDataIn;
        core::CalleeSlot slotVolumetricDataOut;
    };

} /* end namespace astro */
} /* end namespace megamol */

#endif /* MEGAMOL_ASTRO_MINMAX_H_INCLUDED */
