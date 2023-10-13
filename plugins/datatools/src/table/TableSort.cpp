/*
 * TableSort.cpp
 *
 * Copyright (C) 2019 Visualisierungsinstitut der Universität Stuttgart
 * Alle Rechte vorbehalten.
 */

#include "TableSort.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <numeric>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FlexEnumParam.h"


/*
 * megamol::datatools::table::TableSort::TableSort
 */
megamol::datatools::table::TableSort::TableSort()
        : paramColumn("column", "The column to be filtered.")
        , paramIsDescending("descending", "Sort in descending instead of ascending order.")
        , paramIsStable("stableSort", "Use a stable sorting algorithm.") {
    /* Configure and export the parameters. */
    this->paramColumn << new core::param::FlexEnumParam("");
    this->MakeSlotAvailable(&this->paramColumn);

    this->paramIsDescending << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramIsDescending);

    this->paramIsStable << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramIsStable);
}


/*
 * megamol::datatools::table::TableSort::~TableSort
 */
megamol::datatools::table::TableSort::~TableSort() {
    // TODO: this is toxic
    this->Release();
}


/*
 * megamol::datatools::table::TableSort::create
 */
bool megamol::datatools::table::TableSort::create() {
    return true;
}


/*
 * megamol::datatools::table::TableSort::prepareData
 */
bool megamol::datatools::table::TableSort::prepareData(TableDataCall& src, const unsigned int frameID) {
    using namespace core::param;
    using megamol::core::utility::log::Log;

    /* Request the source data. */
    src.SetFrameID(frameID);
    if (!(src) (0)) {
        Log::DefaultLog.WriteError(
            _T("The call to %hs failed in %hs."), TableDataCall::FunctionName(0), TableDataCall::ClassName());
        return false;
    }

    auto isParamsChanged = this->paramColumn.IsDirty() || this->paramColumn.IsDirty() ||
                           this->paramIsDescending.IsDirty() || this->paramIsStable.IsDirty();

    /* (Re-) Generate the data. */
    if (isParamsChanged || (this->inputHash != src.DataHash()) || (this->frameID != src.GetFrameID())) {
        auto column = 0;
        const auto data = src.GetData();
        std::vector<std::size_t> proxy(src.GetRowsCount());

        /* Copy the column descriptors. */
        this->columns.resize(src.GetColumnsCount());
        std::copy(src.GetColumnsInfos(), src.GetColumnsInfos() + this->columns.size(), this->columns.begin());

        /* Update the column selector. */
        {
            auto param = this->paramColumn.Param<FlexEnumParam>();
            param->ClearValues();
            for (auto& c : this->columns) {
                param->AddValue(c.Name());
            }
        }

        /* Determine the index of the reference column. */
        {
            auto c = this->paramColumn.Param<FlexEnumParam>()->Value();
            for (auto& ci : this->columns) {
                if (ci.Name() == c) {
                    break;
                }
                ++column;
            }

            if (column == this->columns.size()) {
                Log::DefaultLog.WriteError("The column \"hs\" cannot be used for "
                                           "sorting, because it does not exist in the source data.",
                    c.c_str());
            }
        }

        /* Sort the index proxy. */
        std::iota(proxy.begin(), proxy.end(), 0);

        const auto isDesc = this->paramIsDescending.Param<BoolParam>()->Value();
        auto pred = [this, column, data, isDesc](const std::size_t l, const std::size_t r) {
            auto lhs = data[l * this->columns.size() + column];
            auto rhs = data[r * this->columns.size() + column];
            return isDesc ? (rhs < lhs) : (lhs < rhs);
        };

        if (this->paramIsStable.Param<BoolParam>()->Value()) {
            std::stable_sort(proxy.begin(), proxy.end(), pred);
        } else {
            std::sort(proxy.begin(), proxy.end(), pred);
        }

        /* Copy the data in sorted order. */
        this->values.resize(src.GetRowsCount() * src.GetColumnsCount());
        auto dst = this->values.data();

        for (auto r : proxy) {
            std::copy(data + r * this->columns.size(), data + (r + 1) * this->columns.size(), dst);
            dst += this->columns.size();
        }

        /* Persist the state of the data. */
        this->frameID = frameID;
        this->inputHash = src.DataHash();

        if (isParamsChanged) {
            ++this->localHash;
            this->paramColumn.ResetDirty();
            this->paramIsDescending.ResetDirty();
            this->paramIsStable.ResetDirty();
        }
    } /* end if (selector || (this->inputHash != src->DataHash()) ... */

    return true;
}


/*
 * megamol::datatools::table::TableSort::release
 */
void megamol::datatools::table::TableSort::release() {}
