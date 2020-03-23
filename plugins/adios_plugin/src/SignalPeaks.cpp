#include "stdafx.h"
#include "SignalPeaks.h"

#include "mmcore/param/IntParam.h"


megamol::adios::SignalPeaks::SignalPeaks()
    : data_out_slot_("dataOut", "Output")
    , data_in_slot_("dataIn", "Input")
    , num_peaks_slot_("numPeaks", "Number of peaks to select")
    , data_hash_(std::numeric_limits<size_t>::max()) {
    data_out_slot_.SetCallback(
        CallADIOSData::ClassName(), CallADIOSData::FunctionName(0), &SignalPeaks::getDataCallback);
    data_out_slot_.SetCallback(
        CallADIOSData::ClassName(), CallADIOSData::FunctionName(1), &SignalPeaks::getHeaderCallback);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<CallADIOSDataDescription>();
    MakeSlotAvailable(&data_in_slot_);

    num_peaks_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&num_peaks_slot_);
}


megamol::adios::SignalPeaks::~SignalPeaks() { this->Release(); }


bool megamol::adios::SignalPeaks::create() { return true; }


void megamol::adios::SignalPeaks::release() {}


bool megamol::adios::SignalPeaks::getDataCallback(core::Call& c) {
    auto out_data = dynamic_cast<CallADIOSData*>(&c);
    if (out_data == nullptr) return false;

    auto in_data = data_in_slot_.CallAs<CallADIOSData>();
    if (in_data == nullptr) return false;

    if (!(*in_data)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("SignalPeaks: Error during GetHeader");
        return false;
    }

    if (data_hash_ != in_data->getDataHash()) {
        data_hash_ = in_data->getDataHash();

        data_map_->clear();

        auto const numPeaks = num_peaks_slot_.Param<core::param::IntParam>()->Value();

        auto avail_vars = in_data->getAvailableVars();

        // iterate through all attributes
        for (auto const& el : avail_vars) {
            in_data->inquire(el);
        }

        if (!(*in_data)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("SignalPeaks: Error during GetData");
            return false;
        }

        std::vector<std::vector<float>> data(avail_vars.size());

        for (size_t i = 0; i < avail_vars.size(); ++i) {
            data[i] = in_data->getData(avail_vars[i])->GetAsFloat();
            if (data[i].size() / 2 < numPeaks) {
                vislib::sys::Log::DefaultLog.WriteError("SignalPeaks: Not enough samples in signal");
                return false;
            }
            std::vector<float> tmp(data[i].size() / 2);
            for (size_t j = 0; j < data[i].size() / 2; ++j) {
                tmp[j] = data[i][j * 2];
            }
            std::nth_element(tmp.begin(), tmp.begin() + numPeaks, tmp.end(), std::greater<float>());
            auto tval = tmp[numPeaks];
            std::stable_partition(data[i].begin(), data[i].end(), [tval](auto const& a) { return a >= tval; });

            auto fCon = std::make_shared<FloatContainer>();
            auto& fVec = fCon->getVec();
            fVec = std::vector<float>(data[i].begin(), data[i].begin() + numPeaks);

            data_map_->operator[](avail_vars[i]) = std::move(fCon);
        }
    }

    out_data->setData(data_map_);

    out_data->setDataHash(data_hash_);

    return true;
}


bool megamol::adios::SignalPeaks::getHeaderCallback(core::Call& c) {
    auto out_data = dynamic_cast<CallADIOSData*>(&c);
    if (out_data == nullptr) return false;

    auto in_data = data_in_slot_.CallAs<CallADIOSData>();
    if (in_data == nullptr) return false;

    if (!(*in_data)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("SignalPeaks: Error during GetHeader");
        return false;
    }

    out_data->setAvailableVars(in_data->getAvailableVars());

    out_data->setFrameCount(in_data->getFrameCount());

    return true;
}
