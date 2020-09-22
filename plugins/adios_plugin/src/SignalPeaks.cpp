#include "stdafx.h"
#include "SignalPeaks.h"

#include <functional>

#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"


megamol::adios::SignalPeaks::SignalPeaks()
    : data_out_slot_("dataOut", "Output")
    , data_in_slot_("dataIn", "Input")
    , num_peaks_slot_("numPeaks", "Number of peaks to select")
    , peak_selector_slot_("peakSelector", "Select peak detection method")
    , column_grouping_factor_slot_("columnGroupingFactor", "Size of column group")
    , data_hash_(std::numeric_limits<size_t>::max()) {
    data_out_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(0), &SignalPeaks::getDataCallback);
    data_out_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(1), &SignalPeaks::getHeaderCallback);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<stdplugin::datatools::table::TableDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    num_peaks_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&num_peaks_slot_);

    auto ep = new core::param::EnumParam(static_cast<int>(PeakSelector::MAXVARIANCE));
    ep->SetTypePair(static_cast<int>(PeakSelector::EQUIDISTANT), "equidistant");
    ep->SetTypePair(static_cast<int>(PeakSelector::MAXVARIANCE), "maxvariance");
    ep->SetTypePair(static_cast<int>(PeakSelector::NTHHIGHEST), "nthhighest");
    peak_selector_slot_ << ep;
    MakeSlotAvailable(&peak_selector_slot_);

    column_grouping_factor_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&column_grouping_factor_slot_);
}


megamol::adios::SignalPeaks::~SignalPeaks() { this->Release(); }


bool megamol::adios::SignalPeaks::create() { return true; }


void megamol::adios::SignalPeaks::release() {}


bool megamol::adios::SignalPeaks::getDataCallback(core::Call& c) {
    auto out_data = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_data == nullptr) return false;

    auto in_data = data_in_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (in_data == nullptr) return false;

    if (!(*in_data)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SignalPeaks: Error during GetHeader");
        return false;
    }

    if (data_hash_ != in_data->DataHash() || isDirty()) {
        data_hash_ = in_data->DataHash();
        resetDirty();

        data_.clear();
        infos_.clear();

        auto const numPeaks = num_peaks_slot_.Param<core::param::IntParam>()->Value();

        if (!(*in_data)(0)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("SignalPeaks: Error during GetData");
            return false;
        }

        auto const num_columns = in_data->GetColumnsCount();
        auto const num_rows = in_data->GetRowsCount();

        auto const in_data_ptr = in_data->GetData();

        out_num_columns_ = num_columns;

        auto const method = peak_selector_slot_.Param<core::param::EnumParam>()->Value();

        std::function<std::vector<size_t>(
            float const* data, size_t num_cols, size_t num_rows, size_t num_peaks)> selector;

        switch (method) {
        case static_cast<int>(PeakSelector::EQUIDISTANT):
            selector = std::bind(&SignalPeaks::equidistant, this, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4);
            break;
        case static_cast<int>(PeakSelector::NTHHIGHEST):
            selector = std::bind(&SignalPeaks::nthhighest, this, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4);
        case static_cast<int>(PeakSelector::MAXVARIANCE):
        default:
            selector = std::bind(&SignalPeaks::maxvariance, this, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, std::placeholders::_4);
        }

        auto const peaks = selector(in_data_ptr, num_columns, num_rows, numPeaks);

        out_num_rows_ = peaks.size();
        data_ = extract_peaks(in_data_ptr, num_columns, num_rows, peaks);
        
        auto const in_infos_ptr = in_data->GetColumnsInfos();

        auto const new_ranges = reevaluate_colums(data_, out_num_columns_, out_num_rows_);

        infos_.resize(num_columns);
        std::copy(in_infos_ptr, in_infos_ptr + num_columns, infos_.data());

        populate_ranges(infos_, new_ranges);
    }

    out_data->Set(out_num_columns_, out_num_rows_, infos_.data(), data_.data());

    out_data->SetDataHash(data_hash_);

    return true;
}


bool megamol::adios::SignalPeaks::getHeaderCallback(core::Call& c) {
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
