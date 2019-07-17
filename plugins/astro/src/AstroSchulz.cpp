/*
 * AstroSchulz.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * All rights reserved.
 */

#include "stdafx.h"
#include "AstroSchulz.h"

#include "vislib/math/mathfunctions.h"


/*
 * megamol::astro::AstroSchulz::AstroSchulz
 */
megamol::astro::AstroSchulz::AstroSchulz(void)
    : Module()
    , frameID(0)
    , hash(0)
    , slotAstroData("astroData", "Input slot for astronomical data")
    , slotTableData("tableData", "Output slot for the resulting sphere data") {
    using megamol::stdplugin::datatools::table::TableDataCall;

    // Connect the slots.
    this->slotAstroData.SetCompatibleCall<AstroDataCallDescription>();
    this->MakeSlotAvailable(&this->slotAstroData);

    this->slotTableData.SetCallback(megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(0), &AstroSchulz::getData);
    this->slotTableData.SetCallback(megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(1), &AstroSchulz::getHash);
    this->MakeSlotAvailable(&this->slotTableData);

    // Define the data format of the output.
    this->columns.reserve(21);

    this->columns.emplace_back();
    this->columns.back().SetName("PositionX");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("PositionY");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("PositionZ");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("VelocityX");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("VelocityY");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("VelocityZ");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("Temperature");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("Mass");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("InternalEnergy");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("SmoothingLength");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("MolecularWeight");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("Density");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("GraviationalPotential");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("Entropy");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("Baryon");
    this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
    this->columns.back().SetMinimumValue(0.0f);
    this->columns.back().SetMaximumValue(1.0f);

    this->columns.emplace_back();
    this->columns.back().SetName("Star");
    this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
    this->columns.back().SetMinimumValue(0.0f);
    this->columns.back().SetMaximumValue(1.0f);

    this->columns.emplace_back();
    this->columns.back().SetName("Wind");
    this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
    this->columns.back().SetMinimumValue(0.0f);
    this->columns.back().SetMaximumValue(1.0f);

    this->columns.emplace_back();
    this->columns.back().SetName("StarFormingGas");
    this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
    this->columns.back().SetMinimumValue(0.0f);
    this->columns.back().SetMaximumValue(1.0f);

    this->columns.emplace_back();
    this->columns.back().SetName("ActiveGalacticNucleus");
    this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
    this->columns.back().SetMinimumValue(0.0f);
    this->columns.back().SetMaximumValue(1.0f);

    this->columns.emplace_back();
    this->columns.back().SetName("ID");
    this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
    this->columns.back().SetMinimumValue(0);
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
}


/*
 * megamol::astro::AstroSchulz::~AstroSchulz
 */
megamol::astro::AstroSchulz::~AstroSchulz(void) { this->Release(); }


/*
 * megamol::astro::AstroSchulz::create
 */
bool megamol::astro::AstroSchulz::create(void) {
    // intentionally empty
    return true;
}


/*
 * megamol::astro::AstroSchulz::release
 */
void megamol::astro::AstroSchulz::release(void) {
    // intentionally empty
}


/*
 * megamol::astro::AstroSchulz::getData
 */
bool megamol::astro::AstroSchulz::getData(core::Call& call) {
    using megamol::stdplugin::datatools::table::TableDataCall;

    auto ast = this->slotAstroData.CallAs<AstroDataCall>();
    auto tab = static_cast<TableDataCall*>(&call);
    auto retval = false;

    if (ast == nullptr) {
        return false;
    }
    if (tab == nullptr) {
        return false;
    }

    auto frameID = tab->GetFrameID();
    ast->SetFrameID(frameID, false);

    if ((*ast)(AstroDataCall::CallForGetData)) {
        if ((this->hash != ast->DataHash()) || (this->frameID == frameID)) {
            this->hash = ast->DataHash();
            this->frameID = frameID;
        }

        auto cnt = ast->GetParticleCount();
        this->values.resize(cnt * this->columns.size());
        auto dst = this->values.data();

        dst = AstroSchulz::convert(dst, this->columns[0], this->columns[1], this->columns[2], ast->GetPositions());
        dst = AstroSchulz::convert(dst, this->columns[3], this->columns[4], this->columns[5], ast->GetVelocities());
        dst = AstroSchulz::convert(dst, this->columns[6], ast->GetTemperature());
        dst = AstroSchulz::convert(dst, this->columns[7], ast->GetMass());
        dst = AstroSchulz::convert(dst, this->columns[8], ast->GetInternalEnergy());
        dst = AstroSchulz::convert(dst, this->columns[9], ast->GetSmoothingLength());
        dst = AstroSchulz::convert(dst, this->columns[10], ast->GetMolecularWeights());
        dst = AstroSchulz::convert(dst, this->columns[11], ast->GetDensity());
        dst = AstroSchulz::convert(dst, this->columns[12], ast->GetGravitationalPotential());
        dst = AstroSchulz::convert(dst, this->columns[13], ast->GetEntropy());
        dst = AstroSchulz::convert(dst, this->columns[14], ast->GetIsBaryonFlags());
        dst = AstroSchulz::convert(dst, this->columns[15], ast->GetIsStarFlags());
        dst = AstroSchulz::convert(dst, this->columns[16], ast->GetIsWindFlags());
        dst = AstroSchulz::convert(dst, this->columns[17], ast->GetIsStarFormingGasFlags());
        dst = AstroSchulz::convert(dst, this->columns[18], ast->GetIsAGNFlags());
        dst = AstroSchulz::convert(dst, this->columns[19], ast->GetParticleIDs());

        tab->SetDataHash(this->hash);
        tab->Set(this->columns.size(), cnt, this->columns.data(), this->values.data());

        retval = true;
    } /* end if ((*ast)(AstroDataCall::CallForGetData)) */

    tab->SetUnlocker(nullptr);
    return retval;
}


/*
 * megamol::astro::AstroSchulz::getHash
 */
bool megamol::astro::AstroSchulz::getHash(core::Call& call) {
    using megamol::stdplugin::datatools::table::TableDataCall;

    auto tab = dynamic_cast<TableDataCall*>(&call);
    if (tab == nullptr) {
        return false;
    }

    //???? this->assertData();

    tab->SetDataHash(this->hash);
    tab->SetUnlocker(nullptr);

    return true;
}


/*
 * megamol::astro::AstroSchulz::convert
 */
float *megamol::astro::AstroSchulz::convert(float *dst, ColumnInfo& ciX,
        ColumnInfo& ciY, ColumnInfo& ciZ, const vec3ArrayPtr& src) {
    assert(dst != nullptr);
    assert(src != nullptr);

    auto rangeX = std::make_pair(
        (std::numeric_limits<float>::max)(),
        std::numeric_limits<float>::lowest());
    auto rangeY = rangeX;
    auto rangeZ = rangeX;

    auto cnt = src->size();

    for (auto s : *src) {
        dst[0 * cnt] = s.x;
        dst[1 * cnt] = s.y;
        dst[2 * cnt] = s.z;

        if (s.x < rangeX.first) {
            rangeX.first = s.x;
        }
        if (s.x > rangeX.second) {
            rangeX.second = s.x;
        }

        if (s.y < rangeY.first) {
            rangeY.first = s.y;
        }
        if (s.y > rangeY.second) {
            rangeY.second = s.y;
        }

        if (s.z < rangeZ.first) {
            rangeZ.first = s.z;
        }
        if (s.z > rangeZ.second) {
            rangeZ.second = s.z;
        }

        assert(rangeX.first <= rangeX.second);
        assert(rangeY.first <= rangeY.second);
        assert(rangeZ.first <= rangeZ.second);
        assert(dst[0 * cnt] >= rangeX.first);
        assert(dst[0 * cnt] <= rangeX.second);
        assert(dst[1 * cnt] >= rangeY.first);
        assert(dst[1 * cnt] <= rangeY.second);
        assert(dst[2 * cnt] >= rangeZ.first);
        assert(dst[2 * cnt] <= rangeZ.second);

        ++dst;
    }

    ciX.SetMinimumValue(rangeX.first);
    ciY.SetMinimumValue(rangeY.first);
    ciZ.SetMinimumValue(rangeZ.first);

    ciX.SetMaximumValue(rangeX.second);
    ciY.SetMaximumValue(rangeY.second);
    ciZ.SetMaximumValue(rangeZ.second);

    dst += 2 * cnt;

    return dst;
}


/*
 * megamol::astro::AstroSchulz::convert
 */
float *megamol::astro::AstroSchulz::convert(float *dst, ColumnInfo& ci,
        const floatArrayPtr& src) {
    assert(dst != nullptr);
    assert(src != nullptr);
    auto range = std::make_pair(
        (std::numeric_limits<float>::max)(),
        std::numeric_limits<float>::lowest());

    for (auto s : *src) {
        *dst++ = s;

        if (s < range.first) {
            range.first = s;
        }
        if (s > range.second) {
            range.second = s;
        }

        assert(range.first <= range.second);
        assert(s >= range.first);
        assert(s <= range.second);
    }

    ci.SetMinimumValue(range.first);
    ci.SetMaximumValue(range.second);

    return dst;
}


/*
 * megamol::astro::AstroSchulz::convert
 */
float *megamol::astro::AstroSchulz::convert(float *dst,
        ColumnInfo& ci, const boolArrayPtr& src) {
    assert(dst != nullptr);
    assert(src != nullptr);

    for (auto s : *src) {
        auto value = s ? 1.0f : 0.0f;
        *dst++ = value;
        assert((value == 0.0f) || (value == 1.0f));
        assert(value >= ci.MinimumValue());
        assert(value <= ci.MaximumValue());
    }

    return dst;
}


/*
 * megamol::astro::AstroSchulz::convert
 */
float *megamol::astro::AstroSchulz::convert(float *dst, ColumnInfo& ci,
        const idArrayPtr& src) {
    assert(dst != nullptr);
    assert(src != nullptr);
    auto range = std::make_pair(
        (std::numeric_limits<float>::max)(),
        std::numeric_limits<float>::lowest());

    for (auto s : *src) {
        auto value = static_cast<float>(s);
        *dst++ = value;

        if (value < range.first) {
            range.first = value;
        }
        if (value > range.second) {
            range.second = value;
        }

        assert(range.first <= range.second);
        assert(value >= range.first);
        assert(value <= range.second);
    }

    ci.SetMinimumValue(range.first);
    ci.SetMaximumValue(range.second);

    return dst;
}
