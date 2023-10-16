/*
 * TableWhere.cpp
 *
 * Copyright (C) 2019 Visualisierungsinstitut der Universität Stuttgart
 * Alle Rechte vorbehalten.
 */

#include "TableWhere.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <numeric>

#include "mmcore/param/BoolParam.h"
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
    NotEqual,
    LowerRange,
    MiddleRange,
    UpperRange,
    LowerPercentile,
    MiddlePercentile,
    UpperPercentile
};


/*
 * megamol::datatools::table::TableWhere::TableWhere
 */
megamol::datatools::table::TableWhere::TableWhere()
        : paramColumn("column", "The column to be filtered.")
        , paramEpsilon("epsilon", "The epsilon value for testing (in-) equality.")
        , paramOperator("operator", "The comparison operator.")
        , paramReference("reference", "The reference value to compare to.")
        , paramUpdateRange("updateRange", "Update the min/max range as the filter changes.") {
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
        param->SetTypePair(Operator::LowerRange, "relative less than");
        param->SetTypePair(Operator::MiddleRange, "around mean");
        param->SetTypePair(Operator::UpperRange, "relative greater than");
        param->SetTypePair(Operator::LowerPercentile, "in bottom percentile");
        param->SetTypePair(Operator::MiddlePercentile, "around median");
        param->SetTypePair(Operator::UpperPercentile, "in top percentile");
        this->paramOperator << param;
        this->MakeSlotAvailable(&this->paramOperator);
    }

    this->paramReference << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->paramReference);

    this->paramUpdateRange << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramUpdateRange);
}


/*
 * megamol::datatools::table::TableWhere::~TableWhere
 */
megamol::datatools::table::TableWhere::~TableWhere() {
    // TODO: this is toxic
    this->Release();
}


/*
 * megamol::datatools::table::TableWhere::create
 */
bool megamol::datatools::table::TableWhere::create() {
    return true;
}


/*
 * megamol::datatools::table::TableWhere::release
 */
void megamol::datatools::table::TableWhere::release() {}


/*
 * megamol::datatools::table::TableWhere::prepareData
 */
bool megamol::datatools::table::TableWhere::prepareData(TableDataCall& src, const unsigned int frameID) {
    using namespace core::param;
    using megamol::core::utility::log::Log;

    /* Request the source data. */
    src.SetFrameID(frameID);
    if (!(src) (0)) {
        Log::DefaultLog.WriteError(
            _T("The call to %hs failed in %hs."), TableDataCall::FunctionName(0), TableDataCall::ClassName());
        return false;
    }

    auto isParamsChanged = this->paramUpdateRange.IsDirty() || this->paramColumn.IsDirty() ||
                           this->paramOperator.IsDirty() || this->paramReference.IsDirty();

    /* (Re-) Generate the data. */
    if (isParamsChanged || (this->inputHash != src.DataHash()) || (this->frameID != src.GetFrameID())) {
        auto column = 0;
        const auto data = src.GetData();
        auto isSort = false;
        std::function<bool(const float)> selector;

        /* Process updates in the configuration. */
        {
            auto c = this->paramColumn.Param<FlexEnumParam>()->Value();
            auto e = this->paramEpsilon.Param<FloatParam>()->Value();
            auto o = this->paramOperator.Param<EnumParam>()->Value();
            auto r = this->paramReference.Param<FloatParam>()->Value();

            this->columns.resize(src.GetColumnsCount());
            std::copy(src.GetColumnsInfos(), src.GetColumnsInfos() + this->columns.size(), this->columns.begin());

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
                auto range = std::make_pair(this->columns[column].MinimumValue(), this->columns[column].MaximumValue());

                switch (o) {
                case Operator::Less:
                    selector = [r](const float v) { return (v < r); };
                    break;

                case Operator::LessOrEqual:
                    selector = [r](const float v) { return (v <= r); };
                    break;

                case Operator::Equal:
                    selector = [r, e](const float v) { return (std::abs(v - r) <= e); };
                    break;

                case Operator::GreaterOrEqual:
                    selector = [r](const float v) { return (v >= r); };
                    break;

                case Operator::Greater:
                    selector = [r](const float v) { return (v > r); };
                    break;

                case Operator::NotEqual:
                    selector = [r, e](const float v) { return (std::abs(v - r) > e); };
                    break;

                case Operator::LowerRange:
                    selector = [r, range](const float v) {
                        assert(range.second >= range.first);
                        auto d = (range.second - range.first) * r;
                        return (v <= (range.first + d));
                    };
                    break;

                case Operator::MiddleRange:
                    selector = [r, range](const float v) {
                        assert(range.second >= range.first);
                        auto d = 1.0f - 0.5f * (range.second - range.first) * r;
                        return ((v >= (range.first + d)) && (v <= (range.second - d)));
                    };
                    break;

                case Operator::UpperRange:
                    selector = [r, range](const float v) {
                        assert(range.second >= range.first);
                        auto d = (range.second - range.first) * r;
                        return (v >= (range.second - d));
                    };
                    break;

                case Operator::LowerPercentile:
                case Operator::MiddlePercentile:
                case Operator::UpperPercentile:
                    isSort = true;
                    break;

                default:
                    Log::DefaultLog.WriteError(_T("The comparison operator %d ")
                                               _T("is unsupported."),
                        o);
                    break;
                }

            } else {
                Log::DefaultLog.WriteWarn(_T("The column \"%hs\" to be filtered ")
                                          _T("was not found in the data set. The %hs module will copy ")
                                          _T("all input rows."),
                    c.c_str(), TableWhere::ClassName());
            }
        }
        assert(((column >= 0) && (column < this->columns.size())) || !selector);

        if (selector || isSort) {
            // Copy selection.
            std::vector<std::size_t> selection;
            selection.reserve(src.GetRowsCount());

            if (selector) {
                // Selection is based on predicate.
                for (auto r = 0; r < src.GetRowsCount(); ++r) {
                    if (selector(data[r * this->columns.size() + column])) {
                        selection.push_back(r);
                    }
                }
            } else {
                // Selection requires sorting.
                const auto o = this->paramOperator.Param<EnumParam>()->Value();
                const auto r = vislib::math::Clamp(this->paramReference.Param<FloatParam>()->Value(), 0.0f, 1.0f);

                selection.resize(src.GetRowsCount());
                std::iota(selection.begin(), selection.end(), 0);

                std::stable_sort(
                    selection.begin(), selection.end(), [this, data, column](const std::size_t l, const std::size_t r) {
                        auto lhs = data[l * this->columns.size() + column];
                        auto rhs = data[r * this->columns.size() + column];
                        return (lhs < rhs);
                    });

                // Compute the number of elements we want to retain.
                const auto cnt = static_cast<std::size_t>(static_cast<double>(r) * src.GetRowsCount());

                switch (o) {
                case Operator::LowerPercentile:
                    // Take first 'cnt' values.
                    selection.resize(cnt);
                    if (!selection.empty()) {
                        Log::DefaultLog.WriteWarn(_T("Selected range is ")
                                                  _T("within [%f, %f]."),
                            data[selection.front() * this->columns.size() + column],
                            data[selection.back() * this->columns.size() + column]);
                    }
                    break;

                case Operator::MiddlePercentile: {
                    auto c = (src.GetRowsCount() - cnt) / 2;
                    selection.erase(selection.begin(), selection.begin() + c);
                    selection.resize(cnt);
                    if (!selection.empty()) {
                        Log::DefaultLog.WriteWarn(_T("Selected range is ")
                                                  _T("within [%f, %f]."),
                            data[selection.front() * this->columns.size() + column],
                            data[selection.back() * this->columns.size() + column]);
                    }
                } break;

                case Operator::UpperPercentile:
                    // Remove everything up to last 'cnt' values.
                    selection.erase(selection.begin(), selection.end() - cnt);
                    if (!selection.empty()) {
                        Log::DefaultLog.WriteWarn(_T("Selected range is ")
                                                  _T("within [%f, %f]."),
                            data[selection.front() * this->columns.size() + column],
                            data[selection.back() * this->columns.size() + column]);
                    }
                    break;

                default:
                    assert(false);
                    break;
                }
            }

            /* Copy the data. */
            this->values.resize(selection.size() * this->columns.size());
            auto d = this->values.data();
            for (auto r : selection) {
                std::copy(data + r * this->columns.size(), data + (r + 1) * this->columns.size(), d);
                d += this->columns.size();
            }

            /* Update the min/max range if requested. */
            if (this->paramUpdateRange.Param<BoolParam>()->Value()) {
                const auto rows = this->values.size() / this->columns.size();

                for (std::size_t c = 0; c < this->columns.size(); ++c) {
                    auto minimum = (std::numeric_limits<float>::max)();
                    auto maximum = (std::numeric_limits<float>::min)();

                    for (std::size_t r = 0; r < rows; ++r) {
                        auto value = this->values[r * this->columns.size() + c];
                        if (value < minimum) {
                            minimum = value;
                        }
                        if (value > maximum) {
                            maximum = value;
                        }

                        this->columns[c].SetMinimumValue(minimum);
                        this->columns[c].SetMaximumValue(maximum);
                    }
                }
            } /* end if (this->paramUpdateRange.Param<BoolParam>()->Value()) */

        } else {
            // Copy everything.
            this->values.resize(src.GetRowsCount() * this->columns.size());
            std::copy(src.GetData(), src.GetData() + this->values.size(), this->values.begin());
        } /* end if (selector || isSort) */

        /* Persist the state of the data. */
        this->frameID = frameID;
        this->inputHash = src.DataHash();

        if (isParamsChanged) {
            ++this->localHash;
            this->paramColumn.ResetDirty();
            this->paramOperator.ResetDirty();
            this->paramReference.ResetDirty();
            this->paramUpdateRange.ResetDirty();
        }
    } /* end if (selector || (this->inputHash != src->DataHash()) ... */

    return true;
}
