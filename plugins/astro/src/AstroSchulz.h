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

#include <array>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "geometry_calls/MultiParticleDataCall.h"

#include "mmcore/param/ParamSlot.h"

#include "datatools/table/TableDataCall.h"


namespace megamol::astro {

/// <summary>
/// Converts from <see cref="AstroDataCall" /> to a table for data
/// visualisation.
/// </summary>
class AstroSchulz : public core::Module {

public:
    static inline const char* ClassName() {
        return "AstroSchulz";
    }

    static inline const char* Description() {
        return "Converts data contained in a AstroDataCall to a "
               "TableDataCall";
    }

    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    AstroSchulz();

    /** Dtor. */
    ~AstroSchulz() override;

protected:
    bool create() override;

    void release() override;

private:
    typedef megamol::datatools::table::TableDataCall::ColumnInfo ColumnInfo;

    static bool getData(AstroDataCall& call, const unsigned int frameID);

    static constexpr inline std::pair<float, float> initialiseRange() {
        return std::make_pair((std::numeric_limits<float>::max)(), std::numeric_limits<float>::lowest());
    }

    static void updateRange(std::pair<float, float>& range, const float value);

    void convert(float* dst, const std::size_t col, const vec3ArrayPtr& src);

    void convert(float* dst, const std::size_t col, const floatArrayPtr& src);

    void convert(float* dst, const std::size_t col, const boolArrayPtr& src);

    void convert(float* dst, const std::size_t col, const idArrayPtr& src);

    bool getData(core::Call& call);

    bool getData(const unsigned int frameID);

    void getData(float* dst, const AstroDataCall& ast);

    inline std::size_t getHash() {
        auto retval = this->hashInput;
        retval ^= this->hashState + 0x9e3779b9 + (retval << 6) + (retval >> 2);
        return retval;
    }

    bool getHash(core::Call& call);

    bool getRanges(const unsigned int start, const unsigned int cnt);

    inline bool isQuantitative(const std::size_t col) {
        using megamol::datatools::table::TableDataCall;
        return ((col < this->columns.size()) && (this->columns[col].Type() == TableDataCall::ColumnType::QUANTITATIVE));
    }

    void norm(float* dst, const std::size_t col, const vec3ArrayPtr& src);

    void setRange(const std::size_t col, const std::pair<float, float>& src);

    std::vector<ColumnInfo> columns;
    unsigned int frameID;
    std::size_t hashInput;
    std::size_t hashState;
    core::param::ParamSlot paramFullRange;
    std::array<core::param::ParamSlot, 27> paramsInclude;
    std::vector<std::pair<float, float>> ranges;
    core::CallerSlot slotAstroData;
    core::CalleeSlot slotTableData;
    std::vector<float> values;
};

} // namespace megamol::astro

#endif /* MEGAMOL_ASTRO_ASTROSCHULZ_H_INCLUDED */
