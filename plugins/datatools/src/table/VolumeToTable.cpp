#include "VolumeToTable.h"

#include "mmcore/utility/log/Log.h"

using namespace megamol::datatools;
using namespace megamol;

/*
 * VolumeToTable::VolumeToTable
 */
VolumeToTable::VolumeToTable(void)
        : Module()
        , slotTableOut("floattable", "Provides the data as table.")
        , slotVolumeIn("particles", "Volume input call") {

    /* Register parameters. */

    /* Register calls. */
    this->slotTableOut.SetCallback(table::TableDataCall::ClassName(), "GetData", &VolumeToTable::getTableData);
    this->slotTableOut.SetCallback(table::TableDataCall::ClassName(), "GetHash", &VolumeToTable::getTableHash);

    this->MakeSlotAvailable(&this->slotTableOut);

    this->slotVolumeIn.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->slotVolumeIn);
}


/*
 * VolumeToTable::~VolumeToTable
 */
VolumeToTable::~VolumeToTable(void) {
    this->Release();
}


/*
 * VolumeToTable::create
 */
bool VolumeToTable::create(void) {
    return true;
}

bool VolumeToTable::assertVDC(geocalls::VolumetricDataCall* in, table::TableDataCall* tc) {

    in->SetFrameID(tc->GetFrameID(), true);
    do {
        if (!(*in)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
            return false;
        if (!(*in)(geocalls::VolumetricDataCall::IDX_GET_DATA))
            return false;
    } while (in->FrameID() != tc->GetFrameID());

    if (in->DataHash() != inHash || in->FrameID() != inFrameID) {
        auto meta = in->GetMetadata();
        auto res = meta->Resolution;
        auto cols = meta->Components;
        num_voxels = res[0] * res[1] * res[2];

        column_infos.clear();

        auto add_col = [&](std::string name, float min, float max) {
            auto& ci = column_infos.emplace_back();
            ci.SetName(name);
            ci.SetType(table::TableDataCall::ColumnType::QUANTITATIVE);
            ci.SetMinimumValue(min);
            ci.SetMaximumValue(max);
        };

        add_col("x", 0, res[0]);
        add_col("y", 0, res[1]);
        add_col("z", 0, res[2]);

        for (auto c = 0; c < cols; ++c) {
            add_col("Comp" + std::to_string(c), meta->MinValues[c], meta->MaxValues[c]);
        }

        everything.resize(num_voxels * column_infos.size());

        for (auto x = 0; x < res[0]; ++x) {
            for (auto y = 0; y < res[1]; ++y) {
                for (auto z = 0; z < res[2]; ++z) {
                    size_t lin_idx = (z * res[1] + y) * res[0] + x;
                    everything[column_infos.size() * lin_idx + 0] = x;
                    everything[column_infos.size() * lin_idx + 1] = y;
                    everything[column_infos.size() * lin_idx + 2] = z;
                    for (auto c = 0; c < cols; ++c) {
                        everything[column_infos.size() * lin_idx + 3 + c] = in->GetAbsoluteVoxelValue(x, y, z, c);
                    }
                }
            }
        }
        inHash = in->DataHash();
        inFrameID = in->FrameID();
    }
    return true;
}

/*
 * VolumeToTable::getTableData
 */
bool VolumeToTable::getTableData(core::Call& call) {
    auto* v = this->slotVolumeIn.CallAs<geocalls::VolumetricDataCall>();

    auto* ft = dynamic_cast<table::TableDataCall*>(&call);
    if (ft == nullptr)
        return false;

    if (v != nullptr) {
        assertVDC(v, ft);
        ft->Set(column_infos.size(), num_voxels, column_infos.data(), everything.data());
    }
    return true;
}


bool VolumeToTable::getTableHash(core::Call& call) {
    auto* v = this->slotVolumeIn.CallAs<geocalls::VolumetricDataCall>();

    auto* ft = dynamic_cast<table::TableDataCall*>(&call);
    if (ft == nullptr)
        return false;

    if (v != nullptr) {
        // TODO is this the right callback?
        if (!(*v)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
            return false;
        ft->SetDataHash(v->DataHash());
        ft->SetFrameCount(v->FrameCount());
        return true;
    }
    return false;
}


/*
 * VolumeToTable::release
 */
void VolumeToTable::release(void) {}
