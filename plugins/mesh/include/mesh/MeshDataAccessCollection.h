/*
 * MeshDataAccessCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include <vector>

#include "mesh.h"

#ifndef MESH_DATA_ACCESS_COLLECTION_H_INCLUDED
#define MESH_DATA_ACCESS_COLLECTION_H_INCLUDED

namespace megamol {
namespace mesh {

class MESH_API MeshDataAccessCollection {
public:

    enum ValueType { BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT, HALF_FLOAT, FLOAT, DOUBLE };

    static constexpr size_t getByteSize(ValueType value_type) {
        size_t retval = 0;

        switch (value_type) {
        case BYTE:
            retval = 1;
            break;
        case UNSIGNED_BYTE:
            retval = 1;
            break;
        case SHORT:
            retval = 2;
            break;
        case UNSIGNED_SHORT:
            retval = 2;
            break;
        case INT:
            retval = 4;
            break;
        case UNSIGNED_INT:
            retval = 4;
            break;
        case HALF_FLOAT:
            retval = 2;
            break;
        case FLOAT:
            retval = 4;
            break;
        case DOUBLE:
            retval = 8;
            break;
        default:
            break;
        }

        return retval;
    }

    struct VertexAttribute {
        uint8_t*     data;
        size_t       byte_size;
        unsigned int component_cnt;
        ValueType    component_type;
        size_t       stride;
        size_t       offset;
    };

    struct IndexData {
        uint8_t*  data;
        size_t    byte_size;
        ValueType type;
    };

    struct Mesh {
        std::vector<VertexAttribute> attributes;
        IndexData                    indices;

        // TODO interleaved flag?
    };

    MeshDataAccessCollection() = default;
    ~MeshDataAccessCollection() = default;

    void addMesh(std::vector<VertexAttribute> const& attribs, IndexData const& indices);
    void addMesh(std::vector<VertexAttribute> && attribs, IndexData const& indices);

    // TODO delete functionality

    std::vector<Mesh>& accessMesh(size_t mesh_idx);

private:

    std::vector<Mesh> meshes;
};

inline void MeshDataAccessCollection::addMesh(std::vector<VertexAttribute> const& attribs, IndexData const& indices) {
    meshes.push_back({attribs, indices});
}

inline void MeshDataAccessCollection::addMesh(std::vector<VertexAttribute>&& attribs, IndexData const& indices) {
    meshes.push_back({attribs, indices});
}

inline std::vector<MeshDataAccessCollection::Mesh>& MeshDataAccessCollection::accessMesh(size_t mesh_idx) {
    return meshes;
}

}
}

#endif // !MESH_DATA_ACCESS_COLLECTION_H_INCLUDED
