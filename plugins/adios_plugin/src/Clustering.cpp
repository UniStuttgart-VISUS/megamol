#include "stdafx.h"
#include "Clustering.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "mmstd_datatools/table/TableDataCall.h"
#include "clustering/DBSCAN2.h"


megamol::adios::Clustering::Clustering()
    : data_out_slot_("dataOut", "Output")
    , data_in_slot_("dataIn", "Input")
    , alg_selector_slot_("algorithm", "Select algorithm for clustering")
    , min_pts_slot_("DBSCAN::minPts", "MinPts")
    , sigma_slot_("DBSCAN::sigma", "Sigma")
    , data_hash_(std::numeric_limits<size_t>::max()) {
    data_out_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(0), &Clustering::getDataCallback);
    data_out_slot_.SetCallback(stdplugin::datatools::table::TableDataCall::ClassName(),
        stdplugin::datatools::table::TableDataCall::FunctionName(1),
        &Clustering::getHeaderCallback);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<stdplugin::datatools::table::TableDataCallDescription>();
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
    auto out_data = dynamic_cast<stdplugin::datatools::table::TableDataCall*>(&c);
    if (out_data == nullptr) return false;

    auto in_data = data_in_slot_.CallAs<stdplugin::datatools::table::TableDataCall>();
    if (in_data == nullptr) return false;

    if (!(*in_data)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Clustering: Error during GetHash");
        return false;
    }

    if (data_hash_ != in_data->DataHash() || isDirty()) {
        data_hash_ = in_data->DataHash();
        resetDirty();

        if (!(*in_data)(0)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("Clustering: Error during GetData");
            return false;
        }

        data_.clear();
        infos_.clear();

        auto const num_columns = in_data->GetColumnsCount();
        auto const num_rows = in_data->GetRowsCount();

        auto const in_data_ptr = in_data->GetData();

        auto points = stdplugin::datatools::clustering::ann_points(num_rows, num_columns, in_data_ptr);

        auto kdtree = ANNkd_tree(points, num_rows, num_columns);

        auto sigma = sigma_slot_.Param<core::param::FloatParam>()->Value();
        auto minpts = min_pts_slot_.Param<core::param::IntParam>()->Value();

        int num_clusters = 0;

        auto clusters = stdplugin::datatools::clustering::DBSCAN_scan(kdtree, sigma * sigma, minpts, num_clusters);

        fillDataVec(clusters, in_data_ptr, num_rows, num_columns);

        auto in_colinfo_ptr = in_data->GetColumnsInfos();

        auto ci = stdplugin::datatools::table::TableDataCall::ColumnInfo();
        ci.SetType(stdplugin::datatools::table::TableDataCall::ColumnType::CATEGORICAL);
        ci.SetName("cluster_id");
        auto min_max = std::minmax_element(clusters.cbegin(), clusters.cend());
        ci.SetMinimumValue(*min_max.first);
        ci.SetMaximumValue(*min_max.second);

        infos_.push_back(ci);
        infos_.insert(infos_.end(), in_colinfo_ptr, in_colinfo_ptr + num_columns);

        out_num_rows_ = num_rows;
        out_num_cols_ = num_columns + 1;
    }

    out_data->Set(out_num_cols_, out_num_rows_, infos_.data(), data_.data());

    out_data->SetDataHash(data_hash_);

    return true;
}


bool megamol::adios::Clustering::getHeaderCallback(core::Call& c) {
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
