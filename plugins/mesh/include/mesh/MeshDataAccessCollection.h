/*
 * MeshDataAccessCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MESH_DATA_ACCESS_COLLECTION_H_INCLUDED
#define MESH_DATA_ACCESS_COLLECTION_H_INCLUDED

#include <vector>
#include "mesh.h"

namespace megamol {
namespace mesh {

class MESH_API MeshDataAccessCollection {
public:
    enum ValueType { BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT, HALF_FLOAT, FLOAT, DOUBLE };
    enum AttributeSemanticType { POSITION, NORMAL, COLOR, TEXCOORD, TANGENT};
    enum PrimitiveType { TRIANGLES, QUADS, LINES };

    static constexpr unsigned int convertToGLType(ValueType value_type) {
        unsigned int retval = 0;

        switch (value_type) {

        case BYTE:
            retval = 0x1400;
            break;
        case UNSIGNED_BYTE:
            retval = 0x1401;
            break;
        case SHORT:
            retval = 0x1402;
            break;
        case UNSIGNED_SHORT:
            retval = 0x1403;
            break;
        case INT:
            retval = 0x1404;
            break;
        case UNSIGNED_INT:
            retval = 0x1405;
            break;
        case HALF_FLOAT:
            retval = 0x140B;
            break;
        case FLOAT:
            retval = 0x1406;
            break;
        case DOUBLE:
            retval = 0x140A;
            break;
        default:
            break;
        }

        return retval;
    }

    static constexpr ValueType covertToValueType(unsigned int gl_type) {
        ValueType retval = BYTE; //TODO default to something more reasonable

        switch (gl_type) {
        case 0x1400:
            retval = BYTE;
            break;
        case 0x1401:
            retval = UNSIGNED_BYTE;
            break;
        case 0x1402:
            retval = SHORT;
            break;
        case 0x1403:
            retval = UNSIGNED_SHORT;
            break;
        case 0x1404:
            retval = INT;
            break;
        case 0x1405:
            retval = UNSIGNED_INT;
            break;
        case 0x140B:
            retval = HALF_FLOAT;
            break;
        case 0x1406:
            retval = FLOAT;
            break;
        case 0x140A:
            retval = DOUBLE;
            break;
        default:
            break;
        }

        return retval;
    }

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
        ValueType component_type;
        size_t stride;
        size_t offset;
        AttributeSemanticType semantic;
    };

    struct IndexData {
        uint8_t*  data;
        size_t    byte_size;
        ValueType type;
    };

    struct Mesh {
        std::vector<VertexAttribute> attributes;
        IndexData                    indices;
        PrimitiveType                primitive_type;

        // TODO interleaved flag?
    };

    MeshDataAccessCollection() = default;
    ~MeshDataAccessCollection() = default;

    void addMesh(std::vector<VertexAttribute> const& attribs, IndexData const& indices, PrimitiveType primitive_type = TRIANGLES);
    void addMesh(std::vector<VertexAttribute>&& attribs, IndexData const& indices, PrimitiveType primitive_type = TRIANGLES);

    // TODO delete functionality

    std::vector<Mesh>& accessMesh();

private:

    std::vector<Mesh> meshes;
};

inline void MeshDataAccessCollection::addMesh(std::vector<VertexAttribute> const& attribs, IndexData const& indices,
    PrimitiveType primitive_type) {
    meshes.push_back({attribs, indices, primitive_type});
}

inline void MeshDataAccessCollection::addMesh(std::vector<VertexAttribute>&& attribs, IndexData const& indices,
    PrimitiveType primitive_type) {
    meshes.push_back({attribs, indices, primitive_type});
}

inline std::vector<MeshDataAccessCollection::Mesh>& MeshDataAccessCollection::accessMesh() {
    return meshes;
}

}
}

#endif // !MESH_DATA_ACCESS_COLLECTION_H_INCLUDED
