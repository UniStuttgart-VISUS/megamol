#include "stdafx.h"
#include "DFT.h"

#include <memory>
#include <vector>

#include "CallADIOSData.h"


megamol::adios::DFT::DFT() : data_out_slot_("dataOut", "Output"), data_in_slot_("dataIn", "Input"), data_hash_(std::numeric_limits<size_t>::max()) {
    data_out_slot_.SetCallback(CallADIOSData::ClassName(), CallADIOSData::FunctionName(0), &DFT::getDataCallback);
    data_out_slot_.SetCallback(CallADIOSData::ClassName(), CallADIOSData::FunctionName(1), &DFT::getHeaderCallback);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<CallADIOSDataDescription>();
    MakeSlotAvailable(&data_in_slot_);
}


megamol::adios::DFT::~DFT() { this->Release(); }


bool megamol::adios::DFT::create() { return true; }


void megamol::adios::DFT::release() {}


bool megamol::adios::DFT::getDataCallback(core::Call& c) {
    auto out_data = dynamic_cast<CallADIOSData*>(&c);
    if (out_data == nullptr) return false;

    auto in_data = data_in_slot_.CallAs<CallADIOSData>();
    if (in_data == nullptr) return false;

    if (!(*in_data)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("DFT: Error during GetHeader");
        return false;
    }

    if (data_hash_ != in_data->getDataHash()) {
        data_hash_ = in_data->getDataHash();

        data_map_->clear();

        auto avail_vars = in_data->getAvailableVars();

        // iterate through all attributes
        for (auto const& el : avail_vars) {
            in_data->inquire(el);
        }

        if (!(*in_data)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("DFT: Error during GetData");
            return false;
        }

        std::vector<std::vector<float>> data(avail_vars.size());
        // std::vector<FFTWArrayC> fft_data;

        // bool is_same_size = true;
        for (size_t i = 0; i < avail_vars.size(); ++i) {
            data[i] = in_data->getData(avail_vars[i])->GetAsFloat();

            auto tmp = FFTWArrayR(data[i].size());
            std::copy(data[i].cbegin(), data[i].cend(), static_cast<float*>(tmp));
            auto out = FFTWArrayC(data[i].size() / 2 + 1);
            FFTWPlan1D(data[i].size(), tmp, out, FFTW_ESTIMATE).Execute();
            // fft_data.push_back(out);

            auto fCon = std::make_shared<FloatContainer>();
            auto& fVec = fCon->getVec();
            fVec.resize((data[i].size() / 2 + 1) * 2);
            for (size_t j = 0; j < data[i].size() / 2 + 1; ++j) {
                fVec[j * 2 + 0] = out[j][0];
                fVec[j * 2 + 1] = out[j][1];
            }

            data_map_->operator[](avail_vars[i]) = std::move(fCon);
        }
    }

    out_data->setData(data_map_);

    out_data->setDataHash(data_hash_);

    return true;
}


bool megamol::adios::DFT::getHeaderCallback(core::Call& c) {
    auto out_data = dynamic_cast<CallADIOSData*>(&c);
    if (out_data == nullptr) return false;

    auto in_data = data_in_slot_.CallAs<CallADIOSData>();
    if (in_data == nullptr) return false;

    if (!(*in_data)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("DFT: Error during GetHeader");
        return false;
    }

    out_data->setAvailableVars(in_data->getAvailableVars());

    out_data->setFrameCount(in_data->getFrameCount());

    return true;
}
