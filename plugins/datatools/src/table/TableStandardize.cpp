/**
 * MegaMol
 * Copyright (c) 2024, MegaMol Dev Team
 * All rights reserved.
 */

#include "TableStandardize.h"

#include <algorithm>
#include <limits>

#include <Eigen/Dense>

#include "mmcore/param/EnumParam.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::datatools;
using namespace megamol::datatools::table;
using namespace megamol;

std::string TableStandardize::ModuleName = std::string("TableStandardize");

template<typename T>
static inline double Lerp(T v0, T v1, T t) {
    return (1 - t) * v0 + t * v1;
}

// https://stackoverflow.com/a/37708864/705750
template<typename T>
static inline std::vector<T> Quantile(const std::vector<T>& inData, const std::vector<T>& probs) {
    if (inData.empty()) {
        return std::vector<T>();
    }

    if (1 == inData.size()) {
        return std::vector<T>(1, inData[0]);
    }

    std::vector<T> data = inData;
    std::sort(data.begin(), data.end());
    std::vector<T> quantiles;

    for (size_t i = 0; i < probs.size(); ++i) {
        T poi = Lerp<T>(-0.5, data.size() - 0.5, probs[i]);

        size_t left = std::max(int64_t(std::floor(poi)), int64_t(0));
        size_t right = std::min(int64_t(std::ceil(poi)), int64_t(data.size() - 1));

        T datLeft = data.at(left);
        T datRight = data.at(right);

        T quantile = Lerp<T>(datLeft, datRight, poi - left);

        quantiles.push_back(quantile);
    }

    return quantiles;
}

TableStandardize::TableStandardize()
        : core::Module()
        , dataOutSlot("dataOut", "Output")
        , dataInSlot("dataIn", "Input")
        , stratSlot("strategy", "which standardization approach to use")
        , frameID(-1)
        , in_datahash(std::numeric_limits<unsigned long>::max())
        , out_datahash(0) {

    this->dataInSlot.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(0), &TableStandardize::processData);
    this->dataOutSlot.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(1), &TableStandardize::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    auto enump = new core::param::EnumParam(Strategies::STANDARD);
    enump->SetTypePair(Strategies::OFF, "Off (nop)");
    enump->SetTypePair(Strategies::STANDARD, "Standard");
    enump->SetTypePair(Strategies::MINMAX, "MinMax");
    enump->SetTypePair(Strategies::MAXABS, "MaxAbs");
    enump->SetTypePair(Strategies::ROBUST, "Robust");
    this->stratSlot << enump;
    this->MakeSlotAvailable(&this->stratSlot);
}

TableStandardize::~TableStandardize() {
    this->Release();
}

bool TableStandardize::create() {
    return true;
}

void TableStandardize::release() {}

bool TableStandardize::processData(core::Call& c) {
    try {
        TableDataCall* outCall = dynamic_cast<TableDataCall*>(&c);
        if (outCall == NULL)
            return false;

        TableDataCall* inCall = this->dataInSlot.CallAs<TableDataCall>();
        if (inCall == NULL)
            return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)())
            return false;

        if (this->in_datahash != inCall->DataHash() || this->frameID != inCall->GetFrameID() ||
            this->AnyParameterDirty()) {
            this->in_datahash = inCall->DataHash();
            this->frameID = inCall->GetFrameID();
            this->ResetAllDirtyFlags();
            this->out_datahash++;

            column_count = inCall->GetColumnsCount();
            column_infos = inCall->GetColumnsInfos();
            row_count = inCall->GetRowsCount();
            in_data = inCall->GetData();

            const auto strat = this->stratSlot.Param<core::param::EnumParam>()->Value();

            this->info.clear();
            this->info.resize(column_count);
            this->data.clear();
            this->data.reserve(column_count * row_count);
            this->data.insert(this->data.end(), &in_data[0], &in_data[column_count * row_count]);

            // eigen would have liked column-major, but...
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> out_mat(
                this->data.data(), row_count, column_count);

            switch (strat) {
            case Strategies::OFF:
                // bypass
                for (int x = 0; x < column_count; ++x) {
                    this->info[x].SetName(column_infos[x].Name());
                }
                break;
            case Strategies::STANDARD: {
                for (int x = 0; x < column_count; ++x) {
                    auto mean = out_mat.col(x).mean();
                    out_mat.col(x) -= Eigen::VectorXf::Constant(row_count, mean);
                    auto std_dev = sqrt(out_mat.col(x).cwiseProduct(out_mat.col(x)).sum() / (row_count - 1));
                    out_mat.col(x) /= std_dev;
                    this->info[x].SetName(column_infos[x].Name() + "_std");
                }
                break;
            }
            case Strategies::MINMAX: {
                for (int x = 0; x < column_count; ++x) {
                    auto min = out_mat.col(x).minCoeff();
                    out_mat.col(x) -= Eigen::VectorXf::Constant(row_count, min);
                    auto max = out_mat.col(x).maxCoeff();
                    out_mat.col(x) / max;
                    this->info[x].SetName(column_infos[x].Name() + "_minmax");
                }
                break;
            }
            case Strategies::MAXABS: {
                for (int x = 0; x < column_count; ++x) {
                    auto maxabs = out_mat.col(x).cwiseAbs().maxCoeff();
                    out_mat.col(x) / maxabs;
                    this->info[x].SetName(column_infos[x].Name() + "_maxabs");
                }
                break;
            }
            case Strategies::ROBUST: {
                for (int x = 0; x < column_count; ++x) {
                    // https://stackoverflow.com/a/62698308/705750
                    // auto copy = out_mat.col(x).replicate(1, 1).reshaped();
                    //std::sort(copy.begin(), copy.end());
                    //auto median =
                    //    copy.size() % 2 == 0 ? copy.segment((copy.size() - 2) / 2, 2).mean() : copy(copy.size() / 2);
                    auto copy = std::vector<float>(out_mat.col(x).data(), out_mat.col(x).data() + row_count);
                    auto quartiles = Quantile<float>(copy, {0.25, 0.5, 0.75});

                    out_mat.col(x) -= Eigen::VectorXf::Constant(row_count, quartiles[1]);
                    out_mat.col(x) /= (quartiles[2] - quartiles[0]);
                    this->info[x].SetName(column_infos[x].Name() + "_robust");
                }
                break;
            }
            }

            for (int x = 0; x < column_count; ++x) {
                this->info[x].SetType(column_infos[x].Type());
                this->info[x].SetMinimumValue(out_mat.col(x).minCoeff());
                this->info[x].SetMaximumValue(out_mat.col(x).maxCoeff());
            }
        }

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetFrameID(this->frameID);
        outCall->SetDataHash(this->out_datahash);

        if (!this->info.empty()) {
            outCall->Set(
                this->info.size(), this->data.size() / this->info.size(), this->info.data(), this->data.data());
        } else {
            outCall->Set(0, 0, NULL, NULL);
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            _T("%hs: Failed to execute processData\n"), ModuleName.c_str());
        return false;
    }

    return true;
}

bool TableStandardize::getExtent(core::Call& c) {
    try {
        TableDataCall* outCall = dynamic_cast<TableDataCall*>(&c);
        if (outCall == NULL)
            return false;

        TableDataCall* inCall = this->dataInSlot.CallAs<TableDataCall>();
        if (inCall == NULL)
            return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)(1))
            return false;

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetDataHash(this->out_datahash); // TODO: this is actually crap if somebody properly checks it
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            _T("Failed to execute %hs::getExtent\n"), ModuleName.c_str());
        return false;
    }

    return true;
}
