/*
 * CallGeneric.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef CALL_GENERIC_H_INCLUDED
#define CALL_GENERIC_H_INCLUDED

#include "mesh.h"
#include "mmcore/AbstractGetDataCall.h"

namespace megamol {
namespace mesh {

template <typename DataType, typename MetaDataType> class MESH_API CallGeneric : public megamol::core::Call 
{
public:
    CallGeneric() = default;
    ~CallGeneric() = default;

    static unsigned int FunctionCount() { return 2; }

    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetData";
        case 1:
            return "GetMetaData";
        }
        return NULL;
    }

    void setData(DataType const& data) { m_data = data; }

    void setMetaData(MetaDataType const& meta_data) { m_meta_data = meta_data; }

    DataType const& getData() { return m_data; }

    MetaDataType const& getMetaData() { return m_meta_data; }

private:
    DataType     m_data;
    MetaDataType m_meta_data;
};

} // namespace mesh
} // namespace megamol

#endif // !CALL_GENERIC_H_INCLUDED
