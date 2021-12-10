/*
 * ParticleDataAccessCollection.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "mmcore/utility/log/Log.h"

namespace megamol::ospray {

class ParticleDataAccessCollection {
public:
    enum ValueType { BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT, HALF_FLOAT, FLOAT, DOUBLE };
    enum AttributeSemanticType { POSITION, INDEX, COLOR, RADIUS, UNKNOWN };

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
        const uint8_t* data;
        size_t byte_size;
        unsigned int component_cnt;
        ValueType component_type;
        size_t stride;
        size_t offset;
        AttributeSemanticType semantic;
    };

    struct Spheres {
        std::vector<VertexAttribute> attributes;
        double global_radius;
        std::array<float, 4> global_color;
    };

    ParticleDataAccessCollection() = default;
    ~ParticleDataAccessCollection() = default;

    void addSphereCollection(std::string const& identifier, std::vector<VertexAttribute> const& attribs,
        double global_radius_ = -1.0, std::array<float, 4> global_color_ = {-1.0f, -1.0f, -1.0f, -1.0f});

    void deleteSphereCollection(std::string const& identifier);

    std::unordered_map<std::string, Spheres>& accessSphereCollections();

    Spheres const& accessSphereCollection(std::string const& identifier);

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
    std::unordered_map<std::string, Spheres> spheres;
};

inline void ParticleDataAccessCollection::addSphereCollection(std::string const& identifier,
    std::vector<VertexAttribute> const& attribs, double global_radius_, std::array<float, 4> global_color_) {
    spheres.insert({identifier, {attribs, global_radius_, global_color_}});
}

inline void ParticleDataAccessCollection::deleteSphereCollection(std::string const& identifier) {
    auto query = spheres.find(identifier);

    if (query != spheres.end()) {
        spheres.erase(query);
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("deleteMesh error: identifier not found.");
    }
}

inline std::unordered_map<std::string, ParticleDataAccessCollection::Spheres>&
ParticleDataAccessCollection::accessSphereCollections() {
    return spheres;
}

inline ParticleDataAccessCollection::Spheres const& ParticleDataAccessCollection::accessSphereCollection(
    std::string const& identifier) {
    auto retval = ParticleDataAccessCollection::Spheres({{}});

    auto query = spheres.find(identifier);

    if (query != spheres.end()) {
        return query->second;
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("accessMesh error: identifier not found.");
    }

    return retval;
}

inline std::vector<std::vector<unsigned int>> ParticleDataAccessCollection::getFormattedAttributeIndices(
    std::string const& identifier) {
    auto retval = std::vector<std::vector<unsigned int>>();

    auto query = spheres.find(identifier);

    if (query != spheres.end()) {
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
        megamol::core::utility::log::Log::DefaultLog.WriteError("accessSphereCollection error: identifier not found.");
    }

    return retval;
}

} // namespace megamol::ospray
