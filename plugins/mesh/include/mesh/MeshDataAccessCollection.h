/*
 * MeshDataAccessCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MESH_DATA_ACCESS_COLLECTION_H_INCLUDED
#define MESH_DATA_ACCESS_COLLECTION_H_INCLUDED

#include <string>
#include <unordered_map>
#include <vector>

#include "mmcore/utility/log/Log.h"

namespace megamol {
namespace mesh {

class MeshDataAccessCollection {
public:
    enum ValueType { BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT, HALF_FLOAT, FLOAT, DOUBLE };
    enum AttributeSemanticType { POSITION, NORMAL, COLOR, TEXCOORD, TANGENT, UNKNOWN, ID };
    enum PrimitiveType { TRIANGLES, QUADS, LINES, LINE_STRIP, TRIANGLE_FAN, TRIANGLE_STRIP };

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
        ValueType retval = BYTE; // TODO default to something more reasonable

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
        uint8_t* data;
        size_t byte_size;
        unsigned int component_cnt;
        ValueType component_type;
        size_t stride;
        size_t offset;
        AttributeSemanticType semantic;
    };

    struct IndexData {
        uint8_t* data;
        size_t byte_size;
        ValueType type;
    };

    struct Mesh {
        std::vector<VertexAttribute> attributes;
        IndexData indices;
        PrimitiveType primitive_type;

        // TODO interleaved flag?
    };

    MeshDataAccessCollection() = default;
    ~MeshDataAccessCollection() = default;

    void addMesh(std::string const& identifier, std::vector<VertexAttribute> const& attribs, IndexData const& indices,
        PrimitiveType primitive_type = TRIANGLES);
    void addMesh(std::string const& identifier, std::vector<VertexAttribute>&& attribs, IndexData const& indices,
        PrimitiveType primitive_type = TRIANGLES);

    void deleteMesh(std::string const& identifier);

    std::unordered_map<std::string, Mesh>& accessMeshes();

    Mesh const& accessMesh(std::string const& identifier);

    /**
     * Get attributes of a mesh grouped by vertex buffer format (non-interleaved vs interleaved mostly).
     * Is computed by checking the data pointers of individual attributes.
     *
     * @param identifier The identifier string of the mesh
     *
     * @return Returns indices into attribute array. Inner vector contains indices of attributes that use the same
     * data buffer. Outer vector contains all groups of indices.
     */
    std::vector<std::vector<unsigned int>> getFormattedAttributeIndices(std::string const& identifier);

private:
    std::unordered_map<std::string, Mesh> meshes;
};

inline void MeshDataAccessCollection::addMesh(std::string const& identifier,
    std::vector<VertexAttribute> const& attribs, IndexData const& indices, PrimitiveType primitive_type) {
    meshes.insert({identifier, {attribs, indices, primitive_type}});
}

inline void MeshDataAccessCollection::addMesh(std::string const& identifier, std::vector<VertexAttribute>&& attribs,
    IndexData const& indices, PrimitiveType primitive_type) {
    meshes.insert({identifier, {attribs, indices, primitive_type}});
}

inline void MeshDataAccessCollection::deleteMesh(std::string const& identifier) {
    auto query = meshes.find(identifier);

    if (query != meshes.end()) {
        meshes.erase(query);
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("deleteMesh error: identifier not found.");
    }
}

inline std::unordered_map<std::string, MeshDataAccessCollection::Mesh>& MeshDataAccessCollection::accessMeshes() {
    return meshes;
}

inline MeshDataAccessCollection::Mesh const& MeshDataAccessCollection::accessMesh(std::string const& identifier) {
    auto retval = MeshDataAccessCollection::Mesh({{}, 0, 0});

    auto query = meshes.find(identifier);

    if (query != meshes.end()) {
        return query->second;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("accessMesh error: identifier not found.");
    }

    return retval;
}

inline std::vector<std::vector<unsigned int>> MeshDataAccessCollection::getFormattedAttributeIndices(
    std::string const& identifier) {
    auto retval = std::vector<std::vector<unsigned int>>();

    auto query = meshes.find(identifier);

    if (query != meshes.end()) {
        for (unsigned int attrib_idx = 0; attrib_idx < query->second.attributes.size(); ++attrib_idx) {
            bool attrib_added = false;
            for (auto& val : retval) {
                if (query->second.attributes[attrib_idx].data == query->second.attributes[val.front()].data) {
                    val.push_back(attrib_idx);
                    attrib_added = true;
                }
            }
            if (!attrib_added) {
                retval.push_back({attrib_idx});
            }
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("accessMesh error: identifier not found.");
    }

    return retval;
}

} // namespace mesh
} // namespace megamol

#endif // !MESH_DATA_ACCESS_COLLECTION_H_INCLUDED
