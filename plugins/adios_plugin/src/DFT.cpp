#include "stdafx.h"
#include "DFT.h"

#include <memory>
#include <vector>


megamol::adios::DFT::DFT()
    : data_out_slot_("dataOut", "Output")
    , data_in_slot_("dataIn", "Input")
    , data_hash_(std::numeric_limits<size_t>::max()) {
    data_out_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(0), &DFT::getDataCallback);
    data_out_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(1), &DFT::getHeaderCallback);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<stdplugin::datatools::table::TableDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);
}


megamol::adios::DFT::~DFT() { this->Release(); }


bool megamol::adios::DFT::create() { return true; }


void megamol::adios::DFT::release() {}


bool megamol::adios::DFT::getDataCallback(core::Call& c) {
    auto out_data = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_data == nullptr) return false;

    auto in_data = data_in_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (in_data == nullptr) return false;

    if (!(*in_data)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("DFT: Error during GetHash");
        return false;
    }

    if (data_hash_ != in_data->DataHash()) {
        data_hash_ = in_data->DataHash();

        data_.clear();
        infos_.clear();

        if (!(*in_data)(0)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("DFT: Error during GetData");
            return false;
        }

        auto const num_columns = in_data->GetColumnsCount();
        auto const num_rows = in_data->GetRowsCount();

        auto const in_data_ptr = in_data->GetData();

        out_num_columns_ = num_columns * 2;
        out_num_rows_ = num_rows / 2 + 1;

        data_.resize(out_num_columns_ * out_num_rows_);
        infos_.resize(out_num_columns_);

        for (size_t col = 0; col < num_columns; ++col) {

            auto tmp = FFTWArrayR(num_rows);
            for (size_t row = 0; row < num_rows; ++row) {
                tmp[row] = in_data->GetData(col, row);
            }

            auto out = FFTWArrayC(out_num_rows_);
            FFTWPlan1D(num_rows, tmp, out, FFTW_ESTIMATE).Execute();

            for (size_t row = 0; row < out_num_rows_; ++row) {
                data_[(col * 2 + 0) + row * out_num_columns_] = out[row][0];
                data_[(col * 2 + 1) + row * out_num_columns_] = out[row][1];
            }
        }

        fillInfoVector(in_data->GetColumnsInfos(), num_columns);
    }

    out_data->Set(out_num_columns_, out_num_rows_, infos_.data(), data_.data());

    out_data->SetDataHash(data_hash_);

    return true;
}


bool megamol::adios::DFT::getHeaderCallback(core::Call& c) {
    auto out_data = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_data == nullptr) return false;

    auto in_data = data_in_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (in_data == nullptr) return false;

    in_data->SetFrameID(out_data->GetFrameID());
    if (!(*in_data)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Clustering: Error during GetHash");
        return false;
    }

    out_data->SetFrameCount(in_data->GetFrameCount());

    out_data->SetFrameID(in_data->GetFrameID());

    return true;
}
