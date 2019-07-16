/*
 * AstroSchulz.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_ASTRO_ASTROSCHULZ_H_INCLUDED
#define MEGAMOL_ASTRO_ASTROSCHULZ_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "astro/AstroDataCall.h"

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"

#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/table/TableDataCall.h"


namespace megamol {
namespace astro {

    /// <summary>
    /// Converts from <see cref="AstroDataCall" /> to a table for data
    /// visualisaion.
    /// </summary>
    class AstroSchulz : public core::Module {

    public:

        static inline const char *ClassName(void) {
            return "AstroSchulz"; 
        }

        static inline const char *Description(void) {
            return "Converts data contained in a AstroDataCall to a MultiParticleDataCall";
        }

        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        AstroSchulz(void);

        /** Dtor. */
        virtual ~AstroSchulz(void);

    protected:

        virtual bool create(void);

        virtual void release(void);

    private:

        typedef megamol::stdplugin::datatools::table::TableDataCall::ColumnInfo
            ColumnInfo;

        static float* convert(float* dst, ColumnInfo& ciX, ColumnInfo& ciY,
            ColumnInfo& ciZ, const vec3ArrayPtr& src);

        static float *convert(float *dst, ColumnInfo& ci,
            const floatArrayPtr& src);

        static float *convert(float *dst, ColumnInfo& ci,
            const boolArrayPtr& src);

        static float *convert(float *dst, ColumnInfo& ci,
            const idArrayPtr& src);

        bool getData(core::Call& call);

        bool getHash(core::Call& call);

        std::vector<ColumnInfo> columns;
        unsigned int frameID;
        std::size_t hash;
        core::CallerSlot slotAstroData;
        core::CalleeSlot slotTableData;
        std::vector<float> values;
    };

} // namespace astro
} // namespace megamol

#endif /* MEGAMOL_ASTRO_ASTROSCHULZ_H_INCLUDED */
