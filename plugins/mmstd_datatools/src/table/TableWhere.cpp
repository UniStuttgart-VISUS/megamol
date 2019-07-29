/*
 * TableWhere.cpp
 *
 * Copyright (C) 2019 Visualisierungsinstitut der Universität Stuttgart
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TableWhere.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"


/// <summary>
/// The list of possible comparison operators.
/// <summary>
enum Operator : int {
    Less = -2,
    LessOrEqual = -1,
    Equal = 0,
    GreaterOrEqual = 1,
    Greater = 2,
    NotEqual = 3
};


/*
 * megamol::stdplugin::datatools::table::TableWhere::TableWhere
 */
megamol::stdplugin::datatools::table::TableWhere::TableWhere(void) : frameID(0),
        inputHash((std::numeric_limits<std::size_t>::max)()),
        localHash((std::numeric_limits<std::size_t>::max)()),
        paramColumn("column", "The column to be filtered."),
        paramEpsilon("epsilon", "The epsilon value for testing (in-) equality."),
        paramOperator("operator", "The comparison operator."),
        paramReference("reference", "The reference value to compare to."),
        slotInput("input", "The input slot providing the unfiltered data."),
        slotOutput("output", "The input slot for the filtered data.") {
    /* Export the calls. */
    this->slotInput.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->slotInput);

    this->slotOutput.SetCallback(TableDataCall::ClassName(),
        TableDataCall::FunctionName(0),
        &TableWhere::getData);
    this->slotOutput.SetCallback(TableDataCall::ClassName(),
        TableDataCall::FunctionName(1),
        &TableWhere::getHash);
    this->MakeSlotAvailable(&this->slotOutput);

    /* Configure and export the parameters. */
    this->paramColumn << new core::param::FlexEnumParam("");
    this->MakeSlotAvailable(&this->paramColumn);

    this->paramEpsilon << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->paramEpsilon);

    {
        auto param = new core::param::EnumParam(0);
        param->SetTypePair(Operator::Less, "less than");
        param->SetTypePair(Operator::LessOrEqual, "less or equal than");
        param->SetTypePair(Operator::Equal, "equals");
        param->SetTypePair(Operator::GreaterOrEqual, "greater or equal than");
        param->SetTypePair(Operator::Greater, "greater than");
        param->SetTypePair(Operator::NotEqual, "does not equal");
        this->paramOperator << param;
        this->MakeSlotAvailable(&this->paramOperator);
    }

    this->paramReference << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->paramReference);
}


/*
 * megamol::stdplugin::datatools::table::TableWhere::~TableWhere
 */
megamol::stdplugin::datatools::table::TableWhere::~TableWhere(void) {
    // TODO: this is toxic
    this->Release();
}


/*
 * megamol::stdplugin::datatools::table::TableWhere::create
 */
bool megamol::stdplugin::datatools::table::TableWhere::create(void) {
    return true;
}


/*
 * megamol::stdplugin::datatools::table::TableWhere::release
 */
void megamol::stdplugin::datatools::table::TableWhere::release(void) { }


/*
 * megamol::stdplugin::datatools::table::TableWhere::getData
 */
bool megamol::stdplugin::datatools::table::TableWhere::getData(
        core::Call& call) {
    using namespace core::param;
    using vislib::sys::Log;

    auto src = this->slotInput.CallAs<TableDataCall>();
    auto dst = dynamic_cast<TableDataCall *>(&call);

    /* Sanity checks. */
    if (src == nullptr) {
        Log::DefaultLog.WriteError(_T("The input slot of %hs is invalid"),
            TableDataCall::ClassName());
        return false;
    }

    if (dst == nullptr) {
        Log::DefaultLog.WriteError(_T("The output slot of %hs is invalid"),
            TableDataCall::ClassName());
        return false;
    }

    /* Request the source data. */
    src->SetFrameID(dst->GetFrameID());
    if (!(*src)(0)) {
        Log::DefaultLog.WriteError(_T("The call to %hs failed in %hs."),
            TableDataCall::FunctionName(0), TableDataCall::ClassName());
        return false;
    }

    auto column = 0;
    std::function<bool(const float)> selector;

    /* Process updates in the configuration. */
    if (this->paramColumn.IsDirty() || this->paramOperator.IsDirty()
            || this->paramReference.IsDirty()) {
        auto c = this->paramColumn.Param<FlexEnumParam>()->Value();
        auto e = this->paramEpsilon.Param<FloatParam>()->Value();
        auto o = this->paramOperator.Param<EnumParam>()->Value();
        auto r = this->paramReference.Param<FloatParam>()->Value();

        this->columns.resize(src->GetColumnsCount());
        std::copy(src->GetColumnsInfos(),
            src->GetColumnsInfos() + this->columns.size(),
            this->columns.begin());

        {
            auto param = this->paramColumn.Param<FlexEnumParam>();
            param->ClearValues();
            for (auto& c : this->columns) {
                param->AddValue(c.Name());
            }
        }

        for (auto& ci : this->columns) {
            if (ci.Name() == c) {
                break;
            }
            ++column;
        }

        if (column != this->columns.size()) {
            switch (o) {
                case Operator::Less:
                    selector = [r](const float v) { return (v < r); };
                    break;

                case Operator::LessOrEqual:
                    selector = [r](const float v) { return (v <= r); };
                    break;

                case Operator::Equal:
                    selector = [r, e](const float v) {
                        return (std::abs(v - r) <= e);
                    };
                    break;

                case Operator::GreaterOrEqual:
                    selector = [r](const float v) { return (v >= r); };
                    break;

                case Operator::Greater:
                    selector = [r](const float v) { return (v > r); };
                    break;

                case Operator::NotEqual:
                    selector = [r, e](const float v) {
                        return (std::abs(v - r) > e);
                    };
                    break;

                default:
                    Log::DefaultLog.WriteError(_T("The comparison operator %d ")
                        _T("is unsupported."), o);
                    break;
            }

        } else {
            Log::DefaultLog.WriteWarn(_T("The column \"%hs\" to be filtered ")
                _T("was not found in the data set. The %hs module will copy ")
                _T("all input rows."), c.c_str(), TableWhere::ClassName());
        }

        this->paramColumn.ResetDirty();
        this->paramOperator.ResetDirty();
        this->paramReference.ResetDirty();

        ++this->localHash;
    }
    assert(((column >= 0) && (column < this->columns.size())) || !selector);

    /* (Re-) Generate the data. */
    if (selector || (this->inputHash != src->DataHash())
            || (this->frameID != src->GetFrameID())) {
        const auto data = src->GetData();

        if (selector) {
            // Copy selection.
            std::vector<std::size_t> selection;
            selection.reserve(src->GetRowsCount());

            for (auto r = 0; r < src->GetRowsCount(); ++r) {
                if (selector(data[r * this->columns.size() + column])) {
                    selection.push_back(r);
                }
            }

            this->values.resize(selection.size() * this->columns.size());
            auto d = this->values.data();
            for (auto r : selection) {
                std::copy(data + r * this->columns.size(),
                    data + (r + 1) * this->columns.size(),
                    d);
                d += this->columns.size();
            }

        } else {
            // Copy everything.
            this->values.resize(src->GetRowsCount() * this->columns.size());
            std::copy(src->GetData(), src->GetData() + this->values.size(),
                this->values.begin());
        }

        this->frameID = dst->GetFrameID();
        this->inputHash = src->DataHash();
    } /* end if (selector || (this->inputHash != src->DataHash()) ... */

    dst->SetFrameCount(src->GetFrameCount());
    dst->SetFrameID(this->frameID);
    dst->SetDataHash(this->getHash());
    dst->Set(this->columns.size(),
        this->values.size() / this->columns.size(),
        this->columns.data(),
        this->values.data());

    //dst->SetUnlocker(nullptr);

    return true;
}


/*
 * megamol::stdplugin::datatools::table::TableWhere::getHash
 */
bool megamol::stdplugin::datatools::table::TableWhere::getHash(
        core::Call& call) {
    using vislib::sys::Log;
    auto src = this->slotInput.CallAs<TableDataCall>();
    auto dst = dynamic_cast<TableDataCall *>(&call);

    /* Sanity checks. */
    if (src == nullptr) {
        Log::DefaultLog.WriteError(_T("The input slot of %hs is invalid"),
            TableDataCall::ClassName());
        return false;
    }

    if (dst == nullptr) {
        Log::DefaultLog.WriteError(_T("The output slot of %hs is invalid"),
            TableDataCall::ClassName());
        return false;
    }

    /* Obtain extents and hash of the source data. */
    src->SetFrameID(dst->GetFrameID());
    if (!(*src)(1)) {
        Log::DefaultLog.WriteError(_T("The call to %hs failed in %hs."),
            TableDataCall::FunctionName(1), TableDataCall::ClassName());
        return false;
    }

    // I don't know why the getHash call passes on the frame count, but it seems
    // to be the expected behaviour ...
    {
        auto cnt = src->GetFrameCount();
        dst->SetFrameCount(cnt);
    }

    dst->SetDataHash(this->getHash());
    dst->SetUnlocker(nullptr);

    return true;
}
