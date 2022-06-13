#include "table/TableSplit.h"

#include "mmcore/param/StringParam.h"


megamol::datatools::table::TableSplit::TableSplit()
        : _selected_data_slot("selectedData", "")
        , _rest_data_slot("restData", "")
        , _in_data_slot("inData", "")
        , _column_selector_slot("columnSelector", "") {
    _selected_data_slot.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(0), &TableSplit::getSelectedDataCB);
    _selected_data_slot.SetCallback(TableDataCall::ClassName(), TableDataCall::FunctionName(1), &TableSplit::getHashCB);
    MakeSlotAvailable(&_selected_data_slot);

    _rest_data_slot.SetCallback(TableDataCall::ClassName(), TableDataCall::FunctionName(0), &TableSplit::getRestDataCB);
    _rest_data_slot.SetCallback(TableDataCall::ClassName(), TableDataCall::FunctionName(1), &TableSplit::getHashCB);
    MakeSlotAvailable(&_rest_data_slot);

    _in_data_slot.SetCompatibleCall<TableDataCallDescription>();
    MakeSlotAvailable(&_in_data_slot);

    _column_selector_slot << new core::param::StringParam("sample_value");
    MakeSlotAvailable(&_column_selector_slot);
}


megamol::datatools::table::TableSplit::~TableSplit() {
    this->Release();
}


bool megamol::datatools::table::TableSplit::create() {
    return true;
}


void megamol::datatools::table::TableSplit::release() {}


bool megamol::datatools::table::TableSplit::getSelectedDataCB(core::Call& c) {
    auto outCall = dynamic_cast<TableDataCall*>(&c);
    if (outCall == nullptr)
        return false;

    auto inCall = _in_data_slot.CallAs<TableDataCall>();
    if (inCall == nullptr)
        return false;

    inCall->SetFrameID(outCall->GetFrameID());
    if (!(*inCall)(0))
        return false;

    if (_in_data_hash != inCall->DataHash() || _frame_id != inCall->GetFrameID() || isDirty()) {
        if (!processData(*inCall))
            return false;

        _in_data_hash = inCall->DataHash();
        _frame_id = inCall->GetFrameID();
        resetDirty();
        ++_out_data_hash;
    }

    outCall->SetFrameCount(inCall->GetFrameCount());
    outCall->SetFrameID(_frame_id);
    outCall->SetDataHash(_out_data_hash);

    outCall->Set(_selected_info.size(), _selected_data.size() / _selected_info.size(), _selected_info.data(),
        _selected_data.data());

    return true;
}


bool megamol::datatools::table::TableSplit::getRestDataCB(core::Call& c) {
    auto outCall = dynamic_cast<TableDataCall*>(&c);
    if (outCall == nullptr)
        return false;

    auto inCall = _in_data_slot.CallAs<TableDataCall>();
    if (inCall == nullptr)
        return false;

    inCall->SetFrameID(outCall->GetFrameID());
    if (!(*inCall)(0))
        return false;

    if (_in_data_hash != inCall->DataHash() || _frame_id != inCall->GetFrameID() || isDirty()) {
        if (!processData(*inCall))
            return false;

        _in_data_hash = inCall->DataHash();
        _frame_id = inCall->GetFrameID();
        resetDirty();
        ++_out_data_hash;
    }

    outCall->SetFrameCount(inCall->GetFrameCount());
    outCall->SetFrameID(_frame_id);
    outCall->SetDataHash(_out_data_hash);

    outCall->Set(_rest_info.size(), _rest_data.size() / _rest_info.size(), _rest_info.data(), _rest_data.data());

    return true;
}


bool megamol::datatools::table::TableSplit::getHashCB(core::Call& c) {
    auto outCall = dynamic_cast<TableDataCall*>(&c);
    if (outCall == nullptr)
        return false;

    auto inCall = _in_data_slot.CallAs<TableDataCall>();
    if (inCall == nullptr)
        return false;

    inCall->SetFrameID(outCall->GetFrameID());
    if (!(*inCall)(1))
        return false;

    outCall->SetFrameCount(inCall->GetFrameCount());
    outCall->SetDataHash(_out_data_hash);

    return true;
}


bool megamol::datatools::table::TableSplit::processData(TableDataCall const& inCall) {
    auto const column_selector = std::string(_column_selector_slot.Param<core::param::StringParam>()->Value());

    auto const infos = inCall.GetColumnsInfos();
    auto const in_data = inCall.GetData();
    auto const column_count = inCall.GetColumnsCount();
    auto const row_count = inCall.GetRowsCount();

    std::vector<size_t> sel_indices;
    sel_indices.reserve(column_count);
    std::vector<size_t> rest_indices;
    rest_indices.reserve(column_count);

    for (size_t idx = 0; idx < column_count; ++idx) {
        if (infos[idx].Name().compare(column_selector) != std::string::npos) {
            sel_indices.push_back(idx);
        } else {
            rest_indices.push_back(idx);
        }
    }

    auto selected_column_count = sel_indices.size();
    auto rest_column_count = rest_indices.size();

    _selected_data.resize(selected_column_count * row_count);
    _rest_data.resize(rest_column_count * row_count);

    _selected_info.resize(selected_column_count);
    _rest_info.resize(rest_column_count);

    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < selected_column_count; ++col) {
            _selected_data[col + row * selected_column_count] = in_data[sel_indices[col] + row * column_count];
        }
        for (size_t col = 0; col < rest_column_count; ++col) {
            _rest_data[col + row * rest_column_count] = in_data[rest_indices[col] + row * column_count];
        }
    }

    for (size_t col = 0; col < selected_column_count; ++col) {
        _selected_info[col] = infos[sel_indices[col]];
    }

    for (size_t col = 0; col < rest_column_count; ++col) {
        _rest_info[col] = infos[rest_indices[col]];
    }

    return true;
}
