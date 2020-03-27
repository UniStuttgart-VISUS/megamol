#include "stdafx.h"
#include "Clustering.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"


megamol::adios::Clustering::Clustering()
    : data_out_slot_("dataOut", "Output")
    , data_in_slot_("dataIn", "Input")
    , alg_selector_slot_("algorithm", "Select algorithm for clustering")
    , min_pts_slot_("DBSCAN::minPts", "MinPts")
    , sigma_slot_("DBSCAN::sigma", "Sigma")
    , data_hash_(std::numeric_limits<size_t>::max()) {
    data_out_slot_.SetCallback(
        CallADIOSData::ClassName(), CallADIOSData::FunctionName(0), &Clustering::getDataCallback);
    data_out_slot_.SetCallback(
        CallADIOSData::ClassName(), CallADIOSData::FunctionName(1), &Clustering::getHeaderCallback);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<CallADIOSDataDescription>();
    MakeSlotAvailable(&data_in_slot_);

    auto ep = new core::param::EnumParam(DBSCAN);
    ep->SetTypePair(DBSCAN, "DBSCAN");
    alg_selector_slot_ << ep;
    alg_selector_slot_.SetUpdateCallback(&Clustering::changeAlgCallback);
    MakeSlotAvailable(&alg_selector_slot_);

    min_pts_slot_ << new core::param::IntParam(1, 1);
    MakeSlotAvailable(&min_pts_slot_);

    sigma_slot_ << new core::param::FloatParam(1.0f, std::numeric_limits<float>::min());
    MakeSlotAvailable(&sigma_slot_);
}


megamol::adios::Clustering::~Clustering() { this->Release(); }


bool megamol::adios::Clustering::create() {
    min_pts_slot_.Param<core::param::IntParam>()->SetGUIVisible(false);
    sigma_slot_.Param<core::param::FloatParam>()->SetGUIVisible(false);

    return true;
}


void megamol::adios::Clustering::release() {}


bool megamol::adios::Clustering::getDataCallback(core::Call& c) {
    auto out_data = dynamic_cast<CallADIOSData*>(&c);
    if (out_data == nullptr) return false;

    auto in_data = data_in_slot_.CallAs<CallADIOSData>();
    if (in_data == nullptr) return false;

    if (!(*in_data)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Clustering: Error during GetHeader");
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
            vislib::sys::Log::DefaultLog.WriteError("Clustering: Error during GetData");
            return false;
        }


    }

    out_data->setData(data_map_);

    out_data->setDataHash(data_hash_);

    return true;
}


bool megamol::adios::Clustering::getHeaderCallback(core::Call& c) {
    auto out_data = dynamic_cast<CallADIOSData*>(&c);
    if (out_data == nullptr) return false;

    auto in_data = data_in_slot_.CallAs<CallADIOSData>();
    if (in_data == nullptr) return false;

    if (!(*in_data)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("Clustering: Error during GetHeader");
        return false;
    }

    out_data->setAvailableVars(in_data->getAvailableVars());

    out_data->setFrameCount(in_data->getFrameCount());

    return true;
}


bool megamol::adios::Clustering::changeAlgCallback(core::param::ParamSlot& p) {
    auto const alg_type = alg_selector_slot_.Param<core::param::EnumParam>()->Value();

    switch (alg_type) {
    case DBSCAN:
    default: {
        min_pts_slot_.Param<core::param::IntParam>()->SetGUIVisible(true);
        sigma_slot_.Param<core::param::FloatParam>()->SetGUIVisible(true);
    }
    }
    return true;
}
