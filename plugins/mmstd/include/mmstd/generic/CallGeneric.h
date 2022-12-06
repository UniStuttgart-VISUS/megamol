/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <type_traits>

#include "mmcore/BoundingBoxes_2.h"
#include "mmstd/data/AbstractGetDataCall.h"

namespace megamol::core {

/**
 * Meta data for spatial 3D data communicates the data bounding box as well as frame count
 * and current frame ID for time dependent data.
 */
struct Spatial3DMetaData {
    unsigned int m_frame_cnt = 0;
    unsigned int m_frame_ID = 0;
    megamol::core::BoundingBoxes_2 m_bboxs;
};

struct EmptyMetaData {};

template<typename DataType, typename MetaDataType>
class GenericCall : public Call {
public:
    using data_type = DataType;
    using meta_data_type = MetaDataType;

    void setData(DataType const& data, uint32_t version) {
        m_data = data;
        m_set_version = version;
    }

    void setMetaData(MetaDataType const& meta_data) {
        m_meta_data = meta_data;
    }

    DataType const& getData() {
        m_get_version = m_set_version;
        return m_data;
    }

    MetaDataType const& getMetaData() {
        return m_meta_data;
    }

    // TODO move setters?

    uint32_t version() {
        return m_set_version;
    }

    bool hasUpdate() {
        return (m_set_version > m_get_version);
    }

private:
    DataType m_data;
    MetaDataType m_meta_data;

    uint32_t m_get_version = 0;
    uint32_t m_set_version = 0;
};

template<typename DataType, typename MetaDataType>
class GenericVersionedCall : public GenericCall<DataType, MetaDataType> {
public:
    static unsigned int FunctionCount() {
        return 2;
    }

    static const unsigned int CallGetData = 0;

    static const unsigned int CallGetMetaData = 1;

    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case CallGetData:
            return "GetData";
        case CallGetMetaData:
            return "GetMetaData";
        }
        return NULL;
    }
};

} // namespace megamol::core
