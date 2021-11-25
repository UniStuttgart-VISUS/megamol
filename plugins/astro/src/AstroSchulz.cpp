/*
 * AstroSchulz.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * All rights reserved.
 */

#include "AstroSchulz.h"
#include "stdafx.h"

#include <array>

#include "mmcore/param/BoolParam.h"

#include "vislib/math/mathfunctions.h"

#include "mmcore/utility/log/Log.h"


/*
 * megamol::astro::AstroSchulz::AstroSchulz
 */
megamol::astro::AstroSchulz::AstroSchulz(void)
        : Module()
        , frameID((std::numeric_limits<unsigned int>::max)())
        , hashInput(0)
        , hashState(0)
        , paramsInclude{{// The ParamSlot has been defeated!!!
              {"includePosition", "Include the position."}, {"includeVelocity", "Include the velocity vectors."},
              {"includeVelocityMagnitude", "Include the magnitude of the velocity vectors."},
              {"includeTemperature", "Include the temperature."}, {"includeMass", "Include the mass."},
              {"includeInternalEnergy", "Include the internal energy."},
              {"includeSmoothingLength", "include the smoothing length."},
              {"includeMolecularWeight", "Include the molecular weight."}, {"includeDensity", "Include the density."},
              {"includeGravitationalPotential", "Include the graviational potential."},
              {"includeEntropy", "Include entropy."}, {"includeBaryon", "Include the Boolean indicating baryons."},
              {"includeStar", "Include the Boolean indicating stars."},
              {"includeWind", "Include the Boolean indicating wind."},
              {"includeStarFormingGas", "Include the Boolean indicating start forming gas."},
              {"includeActiveGalactivNucleus", "Include the Boolean indicating AGNs."},
              {"includeID", "Include the particle ID."},
              {"includeVelocityDerivative", "Include the velocity vector derivatives"},
              {"includeInternalEnergyDerivative", "Include the internal energy derivative"},
              {"includeSmoothingLengthDerivative", "include the smoothing length derivative"},
              {"includeMolecularWeightDerivative", "Include the molecular weight derivative"},
              {"includeDensityDerivative", "Include the density derivative"},
              {"includeGravitationalPotentialDerivative", "Include the graviational potential derivative"},
              {"includeTemperatureDerivative", "Include the temperature derivative"},
              {"includeEntropyDerivative", "Include entropy derivative"},
              {"includeAGNDistances", "Include the distances to the AGNs"},
              {"includeTime", "Load all timesteps into a single table and add the frame number as column."}}}
        , paramFullRange("fullRange", "Scan the whole trajecory for min/man ranges.")
        , slotAstroData("astroData", "Input slot for astronomical data")
        , slotTableData("tableData", "Output slot for the resulting sphere data") {
    // Publish the slots.
    this->slotAstroData.SetCompatibleCall<AstroDataCallDescription>();
    this->MakeSlotAvailable(&this->slotAstroData);

    this->slotTableData.SetCallback(megamol::datatools::table::TableDataCall::ClassName(),
        megamol::datatools::table::TableDataCall::FunctionName(0), &AstroSchulz::getData);
    this->slotTableData.SetCallback(megamol::datatools::table::TableDataCall::ClassName(),
        megamol::datatools::table::TableDataCall::FunctionName(1), &AstroSchulz::getHash);
    this->MakeSlotAvailable(&this->slotTableData);

    this->paramFullRange << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramFullRange);

    for (auto& p : this->paramsInclude) {
        p << new core::param::BoolParam(true);
        this->MakeSlotAvailable(&p);
    }

    this->paramsInclude.back().Param<core::param::BoolParam>()->SetValue(false);
}


/*
 * megamol::astro::AstroSchulz::~AstroSchulz
 */
megamol::astro::AstroSchulz::~AstroSchulz(void) {
    // TODO: This is toxic!
    this->Release();
}


/*
 * megamol::astro::AstroSchulz::create
 */
bool megamol::astro::AstroSchulz::create(void) {
    return true;
}


/*
 * megamol::astro::AstroSchulz::release
 */
void megamol::astro::AstroSchulz::release(void) {}


/*
 * megamol::astro::AstroSchulz::getData
 */
bool megamol::astro::AstroSchulz::getData(AstroDataCall& call, const unsigned int frameID) {
    // Log::DefaultLog.WriteInfo(L"Requesting astro frame %u ...", frameID);
    call.SetFrameID(frameID, true);

    do {
        if (!call(AstroDataCall::CallForGetData)) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("%hs failed in %hs.",
                AstroDataCall::FunctionName(AstroDataCall::CallForGetData), AstroSchulz::ClassName());
            return false;
        }
    } while (call.FrameID() != frameID);

    return true;
}


/*
 * megamol::astro::AstroSchulz::updateRange
 */
void megamol::astro::AstroSchulz::updateRange(std::pair<float, float>& range, const float value) {
    if (value < range.first) {
        range.first = value;
    }
    if (value > range.second) {
        range.second = value;
    }
}


/*
 * megamol::astro::AstroSchulz::convert
 */
void megamol::astro::AstroSchulz::convert(float* dst, const std::size_t col, const vec3ArrayPtr& src) {
    assert(dst != nullptr);
    assert(src != nullptr);

    std::array<std::pair<float, float>, 3> range = {
        AstroSchulz::initialiseRange(), AstroSchulz::initialiseRange(), AstroSchulz::initialiseRange()};

    for (auto s : *src) {
        for (std::size_t i = 0; i < s.length(); ++i) {
            dst[i] = s[i];

            AstroSchulz::updateRange(range[i], dst[i]);
            assert(range[i].first <= range[i].second);
            assert(dst[i] >= range[i].first);
            assert(dst[i] <= range[i].second);
        }

        dst += this->columns.size();
    }

    for (std::size_t i = 0; i < 3; ++i) {
        this->setRange(col + i, range[i]);
    }
}


/*
 * megamol::astro::AstroSchulz::convert
 */
void megamol::astro::AstroSchulz::convert(float* dst, const std::size_t col, const floatArrayPtr& src) {
    assert(dst != nullptr);
    assert(src != nullptr);
    auto range = AstroSchulz::initialiseRange();
    assert(range.first > range.second);

    for (auto s : *src) {
        *dst = s;

        AstroSchulz::updateRange(range, *dst);
        assert(range.first <= range.second);
        assert(s >= range.first);
        assert(s <= range.second);

        dst += this->columns.size();
    }

    this->setRange(col, range);
}


/*
 * megamol::astro::AstroSchulz::convert
 */
void megamol::astro::AstroSchulz::convert(float* dst, const std::size_t col, const boolArrayPtr& src) {
    assert(dst != nullptr);
    assert(src != nullptr);
    auto range = AstroSchulz::initialiseRange();

    for (auto s : *src) {
        *dst = s ? 1.0f : 0.0f;
        assert((*dst == 0.0f) || (*dst == 1.0f));

        AstroSchulz::updateRange(range, *dst);
        assert(range.first <= range.second);
        assert(*dst >= range.first);
        assert(*dst <= range.second);

        dst += this->columns.size();
    }

    // this->columns[col].SetMinimumValue(range.first);
    // this->columns[col].SetMaximumValue(range.second);
}


/*
 * megamol::astro::AstroSchulz::convert
 */
void megamol::astro::AstroSchulz::convert(float* dst, const std::size_t col, const idArrayPtr& src) {
    assert(dst != nullptr);
    assert(src != nullptr);
    auto range = AstroSchulz::initialiseRange();

    for (auto s : *src) {
        *dst = static_cast<float>(s);

        AstroSchulz::updateRange(range, *dst);
        assert(range.first <= range.second);
        assert(*dst >= range.first);
        assert(*dst <= range.second);

        dst += this->columns.size();
    }

    this->setRange(col, range);
}


/*
 * megamol::astro::AstroSchulz::getData
 */
bool megamol::astro::AstroSchulz::getData(core::Call& call) {
    using megamol::core::utility::log::Log;
    using megamol::datatools::table::TableDataCall;

    auto ast = this->slotAstroData.CallAs<AstroDataCall>();
    auto tab = static_cast<TableDataCall*>(&call);

    if (ast == nullptr) {
        Log::DefaultLog.WriteWarn("AstroDataCall is not connected "
                                  "in AstroSchulz.",
            nullptr);
        return false;
    }
    if (tab == nullptr) {
        Log::DefaultLog.WriteWarn("TableDataCall is not connected "
                                  "in AstroSchulz.",
            nullptr);
        return false;
    }

    if (!(*ast)(AstroDataCall::CallForGetExtent)) {
        Log::DefaultLog.WriteWarn("AstroDataCall::CallForGetExtent failed "
                                  "in AstroSchulz.",
            nullptr);
        return false;
    }

    if (this->paramFullRange.IsDirty()) {
        auto p = this->paramFullRange.Param<core::param::BoolParam>();

        if (p->Value()) {
            this->getRanges(0, ast->FrameCount());
        } else {
            this->ranges.clear();
        }

        this->paramFullRange.ResetDirty();
    }

    if (!this->getData(tab->GetFrameID())) {
        return false;
    }

    tab->SetFrameCount(ast->FrameCount());
    tab->SetDataHash(this->getHash());
    tab->Set(
        this->columns.size(), this->values.size() / this->columns.size(), this->columns.data(), this->values.data());
    tab->SetUnlocker(nullptr);

    return true;
}


/*
 * megamol::astro::AstroSchulz::getData
 */
bool megamol::astro::AstroSchulz::getData(const unsigned int frameID) {
    using namespace core::param;
    using megamol::core::utility::log::Log;
    using megamol::datatools::table::TableDataCall;

    auto ast = this->slotAstroData.CallAs<AstroDataCall>();
    if (ast == nullptr) {
        Log::DefaultLog.WriteWarn("AstroDataCall is not connected "
                                  "in AstroSchulz.",
            nullptr);
        return false;
    }

    auto isParamChange = this->columns.empty() || std::any_of(this->paramsInclude.begin(), this->paramsInclude.end(),
                                                      [](const ParamSlot& param) { return param.IsDirty(); });

    // Update column metadata if selection changed.
    if (isParamChange) {
        auto col = 0;
        this->columns.clear();
        this->columns.reserve(21);

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
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
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
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
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("Velocity");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("Temperature");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("Mass");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("InternalEnergy");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("SmoothingLength");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("MolecularWeight");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("Density");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("GraviationalPotential");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("Entropy");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("Baryon");
            this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
            this->columns.back().SetMinimumValue(0.0f);
            this->columns.back().SetMaximumValue(1.0f);
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("Star");
            this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
            this->columns.back().SetMinimumValue(0.0f);
            this->columns.back().SetMaximumValue(1.0f);
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("Wind");
            this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
            this->columns.back().SetMinimumValue(0.0f);
            this->columns.back().SetMaximumValue(1.0f);
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("StarFormingGas");
            this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
            this->columns.back().SetMinimumValue(0.0f);
            this->columns.back().SetMaximumValue(1.0f);
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("ActiveGalacticNucleus");
            this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
            this->columns.back().SetMinimumValue(0.0f);
            this->columns.back().SetMaximumValue(1.0f);
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("ID");
            this->columns.back().SetType(TableDataCall::ColumnType::CATEGORICAL);
            this->columns.back().SetMinimumValue(0.0f);
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("VelocityDerivativeX");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

            this->columns.emplace_back();
            this->columns.back().SetName("VelocityDerivativeY");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());

            this->columns.emplace_back();
            this->columns.back().SetName("VelocityDerivativeZ");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("InternalEnergyDerivative");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("SmoothingLengthDerivative");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("MolecularWeightDerivative");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("DensityDerivative");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("GraviationalPotentialDerivative");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("TemperatureDerivative");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("EntropyDerivative");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("AGNDistances");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(std::numeric_limits<float>::lowest());
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        // Further values have to be inserted here. This always has to happen before the frame column is set.

        if (this->paramsInclude[col++].Param<BoolParam>()->Value()) {
            this->columns.emplace_back();
            this->columns.back().SetName("Frame");
            this->columns.back().SetType(TableDataCall::ColumnType::QUANTITATIVE);
            this->columns.back().SetMinimumValue(0.0f);
            this->columns.back().SetMaximumValue((std::numeric_limits<float>::max)());
        }

        for (auto& p : this->paramsInclude) {
            p.ResetDirty();
        }
    }

    if (this->paramsInclude.back().Param<BoolParam>()->Value()) {
        Log::DefaultLog.WriteInfo("Creating union of all astro frames ...");

        if (!(*ast)(AstroDataCall::CallForGetExtent)) {
            Log::DefaultLog.WriteInfo("Failed to get extents of astro data.");
            return false;
        }

        if (isParamChange) {
            this->frameID = (std::numeric_limits<decltype(this->frameID)>::max)();
            this->hashInput = (std::numeric_limits<decltype(this->hashInput)>::max)();
            ++this->hashState;

            this->values.clear();

            const auto cntFrames = ast->FrameCount();
            for (auto frameID = 0; frameID < cntFrames; ++frameID) {
                Log::DefaultLog.WriteInfo("Adding astro frame %u to the union.", frameID);
                if (!AstroSchulz::getData(*ast, frameID)) {
                    return false;
                }

                // Determine location and size of current frame.
                const auto cnt = ast->GetParticleCount();
                const auto offset = this->values.size();
                const auto size = offset + cnt * this->columns.size();

                // If this is the first frame, reserve memory for all.
                if (offset == 0) {
                    this->values.reserve(cntFrames * size);
                }

                // Actually resize the table to hold the current frame.
                this->values.resize(size);

                // Add the current frame at the end.
                this->getData(this->values.data() + offset, *ast);

                // Add the frame number.
                for (auto r = 0; r < cnt; ++r) {
                    this->values[offset + r * this->columns.size() + (this->columns.size() - 1)] = frameID;
                }
            }
        }

    } else {
        // Receive a single frame into 'ast'.
        if (!AstroSchulz::getData(*ast, frameID)) {
            return false;
        }

        // Copy the data into the table as necessary.
        bool hashChanged = this->hashInput != ast->DataHash();
        if (isParamChange || hashChanged || (this->frameID != frameID)) {
            Log::DefaultLog.WriteInfo("Astro data are new (frame %u), filling "
                                      "table ...",
                frameID);
            this->frameID = frameID;
            this->hashInput = ast->DataHash();
            if (isParamChange || hashChanged) {
                ++this->hashState;
            }

            const auto cnt = ast->GetParticleCount();
            this->values.resize(cnt * this->columns.size());
            this->getData(this->values.data(), *ast);
        }
    }

    return true;
}


/*
 * megamol::astro::AstroSchulz::getData
 */
void megamol::astro::AstroSchulz::getData(float* dst, const AstroDataCall& ast) {
    using core::param::BoolParam;

    auto col = 0;
    auto cnt = ast.GetParticleCount();
    auto logicalCol = 0;

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetPositions());
        col += 3;
        dst += 3;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetVelocities());
        col += 3;
        dst += 3;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->norm(dst, col, ast.GetVelocities());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetTemperature());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetMass());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetInternalEnergy());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetSmoothingLength());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetMolecularWeights());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetDensity());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetGravitationalPotential());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetEntropy());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetIsBaryonFlags());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetIsStarFlags());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetIsWindFlags());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetIsStarFormingGasFlags());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetIsAGNFlags());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetParticleIDs());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetVelocityDerivatives());
        col += 3;
        dst += 3;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetInternalEnergyDerivatives());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetSmoothingLengthDerivatives());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetMolecularWeightDerivatives());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetDensityDerivative());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetGravitationalPotentialDerivatives());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetTemperatureDerivatives());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetEntropyDerivatives());
        ++col;
        ++dst;
    }

    if (this->paramsInclude[logicalCol++].Param<BoolParam>()->Value()) {
        this->convert(dst, col, ast.GetAgnDistances());
        ++col;
        ++dst;
    }

    // new values have to be included here, using the correct order
}


/*
 * megamol::astro::AstroSchulz::getHash
 */
bool megamol::astro::AstroSchulz::getHash(core::Call& call) {
    using megamol::datatools::table::TableDataCall;

    auto ast = this->slotAstroData.CallAs<AstroDataCall>();
    auto tab = dynamic_cast<TableDataCall*>(&call);
    if (tab == nullptr) {
        return false;
    }
    if (ast == nullptr) {
        return false;
    }

    //???? this->assertData();
    if (!(*ast)(AstroDataCall::CallForGetExtent)) {
        return false;
    }

    auto fc = ast->FrameCount();
    tab->SetFrameCount(ast->FrameCount());

    tab->SetDataHash(this->getHash());
    tab->SetUnlocker(nullptr);

    return true;
}


/*
 * megamol::astro::AstroSchulz::getRanges
 */
bool megamol::astro::AstroSchulz::getRanges(const unsigned int start, const unsigned int cnt) {
    using megamol::core::utility::log::Log;

    // Note: option to disable column implies that the columns are not generated
    // before the first data are retrieved. Therefore, we need to test-retrieve
    // the first frame here to find out the number of columns.
    decltype(this->ranges) ranges;
    if (this->getData(start)) {
        ranges.resize(this->columns.size());
    }

    Log::DefaultLog.WriteInfo("Scanning astro trajectory for global "
                              "min/max ranges. Please wait while MegaMol is working for you ...",
        nullptr);

    std::generate(ranges.begin(), ranges.end(), AstroSchulz::initialiseRange);
    this->ranges.clear();

    for (unsigned int f = start; f < start + cnt; ++f) {
        if (this->getData(f)) {
            for (std::size_t c = 0; c < this->columns.size(); ++c) {
                AstroSchulz::updateRange(ranges[c], this->columns[c].MinimumValue());
                AstroSchulz::updateRange(ranges[c], this->columns[c].MaximumValue());
            }
        } else {
            return false;
        }
    }

    this->ranges = std::move(ranges);

    for (std::size_t c = 0; c < this->columns.size(); ++c) {
        if (this->isQuantitative(c)) {
            Log::DefaultLog.WriteInfo("Values of column \"%hs\" are within "
                                      "[%f, %f].",
                this->columns[c].Name().c_str(), this->ranges[c].first, this->ranges[c].second);
        }
    }

    return true;
}


/*
 * megamol::astro::AstroSchulz::norm
 */
void megamol::astro::AstroSchulz::norm(float* dst, const std::size_t col, const vec3ArrayPtr& src) {
    auto range = AstroSchulz::initialiseRange();

    for (auto s : *src) {
        *dst = glm::length(s);

        AstroSchulz::updateRange(range, *dst);
        assert(range.first <= range.second);
        assert(*dst >= range.first);
        assert(*dst <= range.second);

        dst += this->columns.size();
    }

    this->setRange(col, range);
}


/*
 * megamol::astro::AstroSchulz::setRange
 */
void megamol::astro::AstroSchulz::setRange(const std::size_t col, const std::pair<float, float>& src) {
    assert(col < this->columns.size());
    assert(src.first <= src.second);
    if (this->ranges.empty()) {
        this->columns[col].SetMinimumValue(src.first);
        this->columns[col].SetMaximumValue(src.second);
    } else if (this->isQuantitative(col)) {
        this->columns[col].SetMinimumValue(this->ranges[col].first);
        this->columns[col].SetMaximumValue(this->ranges[col].second);
    }
}
