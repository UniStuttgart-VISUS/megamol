#include "TableItemSelector.h"

#include <functional>

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"

#include "datatools/table/TableDataCall.h"


megamol::datatools::table::TableItemSelector::TableItemSelector()
        : data_out_slot_("dataOut", "Output")
        , data_in_slot_("dataIn", "Input")
        , selection_column_slot_("selectionColumn", "Column that defines the selection")
        , selection_algorithm_slot_("selectionAlgorithm", "Algorithm for selection") {
    data_out_slot_.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(0), &TableItemSelector::getDataCallback);
    data_out_slot_.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(1), &TableItemSelector::getHashCallback);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<TableDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    selection_column_slot_ << new core::param::FlexEnumParam("undef");
    MakeSlotAvailable(&selection_column_slot_);

    auto ep = new core::param::EnumParam(static_cast<int>(SELECT_ALG::MIN_DIFF));
    ep->SetTypePair(static_cast<int>(SELECT_ALG::FIRST_OF_ALL), "FIRST_OF_ALL");
    ep->SetTypePair(static_cast<int>(SELECT_ALG::MIN_DIFF), "MIN_DIFF");
    selection_algorithm_slot_ << ep;
    MakeSlotAvailable(&selection_algorithm_slot_);
}


megamol::datatools::table::TableItemSelector::~TableItemSelector() {
    this->Release();
}


bool megamol::datatools::table::TableItemSelector::create() {
    return true;
}


void megamol::datatools::table::TableItemSelector::release() {}


bool megamol::datatools::table::TableItemSelector::getDataCallback(core::Call& c) {
    auto out_data = dynamic_cast<datatools::table::TableDataCall*>(&c);
    if (out_data == nullptr)
        return false;

    auto in_data = data_in_slot_.CallAs<datatools::table::TableDataCall>();
    if (in_data == nullptr)
        return false;

    if (!(*in_data)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("SignalPeaks: Error during GetHeader");
        return false;
    }

    if (data_hash_ != in_data->DataHash() || is_dirty()) {
        data_hash_ = in_data->DataHash();
        reset_dirty();

        if (!(*in_data)(0)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("SignalPeaks: Error during GetData");
            return false;
        }

        auto const num_columns = in_data->GetColumnsCount();
        auto const num_rows = in_data->GetRowsCount();

        auto const in_data_ptr = in_data->GetData();
        auto const in_infos_ptr = in_data->GetColumnsInfos();

        fill_enum(in_infos_ptr, num_columns);

        auto const selection_column_name = selection_column_slot_.Param<core::param::FlexEnumParam>()->Value();

        auto const it = column_map_.find(selection_column_name);

        if (it != column_map_.end()) {
            data_.clear();
            infos_.clear();

            auto const selection_column_idx = it->second;

            auto const selection_column = extract_column(in_data_ptr, num_columns, num_rows, selection_column_idx);

            auto const method = selection_algorithm_slot_.Param<core::param::EnumParam>()->Value();

            std::function<std::vector<size_t>(float const* data, size_t num_columns, size_t num_rows,
                std::vector<float> const& selector, size_t selector_idx)>
                selector;

            switch (method) {
            case static_cast<int>(SELECT_ALG::FIRST_OF_ALL):
                selector = std::bind(&TableItemSelector::first_of_all, this, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
                break;
            case static_cast<int>(SELECT_ALG::MIN_DIFF):
            default:
                selector = std::bind(&TableItemSelector::min_difference, this, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
            }

            auto const indices = selector(in_data_ptr, num_columns, num_rows, selection_column, selection_column_idx);

            out_num_rows_ = indices.size();
            out_num_columns_ = num_columns;
            data_ = extract_rows(in_data_ptr, num_columns, num_rows, indices);
            infos_.resize(num_columns);
            for (size_t col = 0; col < num_columns; ++col) {
                infos_[col] = in_infos_ptr[col];
            }
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("TableItemSelector: Column not found");
        }
    }

    out_data->Set(out_num_columns_, out_num_rows_, infos_.data(), data_.data());

    out_data->SetDataHash(data_hash_);

    return true;
}


bool megamol::datatools::table::TableItemSelector::getHashCallback(core::Call& c) {
    auto out_data = dynamic_cast<datatools::table::TableDataCall*>(&c);
    if (out_data == nullptr)
        return false;

    auto in_data = data_in_slot_.CallAs<datatools::table::TableDataCall>();
    if (in_data == nullptr)
        return false;

    in_data->SetFrameID(out_data->GetFrameID());
    if (!(*in_data)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("TableItemSelector: Error during GetHash");
        return false;
    }

    out_data->SetFrameCount(in_data->GetFrameCount());

    out_data->SetFrameID(in_data->GetFrameID());

    return true;
}
