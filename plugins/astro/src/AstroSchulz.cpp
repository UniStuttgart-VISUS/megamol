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

    this->slotTableData.SetCallback(
        megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(0),
        &AstroSchulz::getData);
    this->slotTableData.SetCallback(
        megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(1),
        &AstroSchulz::getHash);
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
    this->columns.back().SetName("Internal energy");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("Smoothing length");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("Molecular weight");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("Density");
    this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
    this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
    this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

    this->columns.emplace_back();
    this->columns.back().SetName("Graviational potential");
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
    this->columns.back().SetName("Star-forming gas");
    this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
    this->columns.back().SetMinimumValue(0.0f);
    this->columns.back().SetMaximumValue(1.0f);

    this->columns.emplace_back();
    this->columns.back().SetName("Active galactic nucleus");
    this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
    this->columns.back().SetMinimumValue(0.0f);
    this->columns.back().SetMaximumValue(1.0f);

    //this->columns.emplace_back();
    //this->columns.back().SetName("ID");
    //this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
    //this->columns.back().SetMinimumValue(0);
    //this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
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
        //dst = AstroSchulz::convert(dst, this->columns[19], ast->GetParticleIDs());

        tab->SetDataHash(this->hash);
        tab->Set(this->columns.size(), cnt, this->columns.data(),
            this->values.data());

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

    auto minX = (std::numeric_limits<float>::max)();
    auto minY = (std::numeric_limits<float>::max)();
    auto minZ = (std::numeric_limits<float>::max)();

    auto maxX = (std::numeric_limits<float>::lowest)();
    auto maxY = (std::numeric_limits<float>::lowest)();
    auto maxZ = (std::numeric_limits<float>::lowest)();

    auto cnt = src->size();

    for (auto s : *src) {
        dst[0] = s.x;
        dst[cnt] = s.y;
        dst[2 * cnt] = s.z;

        if (s.x < minX) {
            minX = s.x;
        }
        if (s.y < minY) {
            minY = s.y;
        }
        if (s.z < minZ) {
            minZ = s.z;
        }

        if (s.x > maxX) {
            maxX = s.x;
        }
        if (s.y > maxY) {
            maxY = s.y;
        }
        if (s.z > maxZ) {
            maxZ = s.z;
        }

        ++dst;
    }

    ciX.SetMinimumValue(minX);
    ciY.SetMinimumValue(minY);
    ciZ.SetMinimumValue(minZ);

    ciX.SetMaximumValue(maxX);
    ciY.SetMaximumValue(maxY);
    ciZ.SetMaximumValue(maxZ);

    dst += 2 * cnt;

    return dst;
}


/*
 * megamol::astro::AstroSchulz::convert
 */
float *megamol::astro::AstroSchulz::convert(float *dst, ColumnInfo& ci,
        const floatArrayPtr &src) {
    assert(dst != nullptr);
    assert(src != nullptr);
    auto i = (std::numeric_limits<float>::max)();
    auto a = (std::numeric_limits<float>::lowest)();

    for (auto s : *src) {
        *dst++ = s;

        if (s < i) {
            i = s;
        }
        if (s > a) {
            a = s;
        }
    }

    ci.SetMinimumValue(i);
    ci.SetMaximumValue(a);

    return dst;
}


/*
 * megamol::astro::AstroSchulz::convert
 */
float *megamol::astro::AstroSchulz::convert(float *dst, ColumnInfo& ci,
        const boolArrayPtr& src) {
    assert(dst != nullptr);
    assert(src != nullptr);
    auto i = (std::numeric_limits<float>::max)();
    auto a = (std::numeric_limits<float>::lowest)();

    for (auto s : *src) {
        auto t = s ? 1.0f : 0.0f;
        *dst++ = t;

        if (t < i) {
            i = t;
        }
        if (t > a) {
            a = t;
        }
    }

    //ci.SetMinimumValue(i);
    //ci.SetMaximumValue(a);

    return dst;
}


/*
 * megamol::astro::AstroSchulz::convert
 */
float *megamol::astro::AstroSchulz::convert(float *dst, ColumnInfo& ci,
        const idArrayPtr& src) {
    assert(dst != nullptr);
    assert(src != nullptr);
    auto i = (std::numeric_limits<float>::max)();
    auto a = (std::numeric_limits<float>::lowest)();

    for (auto s : *src) {
        auto t = static_cast<float>(s);
        *dst++ = t;

        if (t < i) {
            i = t;
        }
        if (t > a) {
            a = t;
        }
    }

    ci.SetMinimumValue(i);
    ci.SetMaximumValue(a);

    return dst;
}
