/*
 * STLDataSource.cpp
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "datatools_gl/io/STLDataSource.h"

#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/Call.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/DataHash.h"

#include "geometry_calls_gl/CallTriMeshDataGL.h"

#include "mmcore/utility/log/Log.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace megamol {
namespace datatools_gl {
namespace io {
STLDataSource::STLDataSource()
        : filename_slot("STL file", "The name of to the STL file to load")
        , mesh_output_slot("mesh_data", "Slot to request mesh data")
{
    // Create file name textbox
    this->filename_slot << new core::param::FilePathParam("");
    Module::MakeSlotAvailable(&this->filename_slot);

    // Create output slot for triangle mesh data
    this->mesh_output_slot.SetCallback(
        geocalls_gl::CallTriMeshDataGL::ClassName(), "GetExtent", &STLDataSource::get_extent_callback);
    this->mesh_output_slot.SetCallback(
        geocalls_gl::CallTriMeshDataGL::ClassName(), "GetData", &STLDataSource::get_mesh_data_callback);

    Module::MakeSlotAvailable(&this->mesh_output_slot);

    // Create resources
    this->vertex_buffer = std::make_shared<std::vector<float>>();
    this->normal_buffer = std::make_shared<std::vector<float>>();
    this->index_buffer = std::make_shared<std::vector<unsigned int>>();
}

STLDataSource::~STLDataSource() {
    this->Release();
}

bool STLDataSource::create() {
    return true;
}

bool STLDataSource::get_input() {
    if (this->filename_slot.IsDirty()) {
        this->filename_slot.ResetDirty();

        // Read data
        const auto& filepath = this->filename_slot.Param<core::param::FilePathParam>()->Value();
        const std::string filename(filepath.generic_u8string());

        try {
            read(filename);
        } catch (const std::runtime_error& ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Request for extent of plugin 'STL Reader' failed: %s", ex.what());

            return false;
        }

        // Extract extent information
        this->min_x = this->min_y = this->min_z = std::numeric_limits<float>::max();
        this->max_x = this->max_y = this->max_z = std::numeric_limits<float>::lowest();

        const auto& vertices = *this->vertex_buffer;

        for (std::size_t value_index = 0; value_index < static_cast<std::size_t>(this->num_triangles * 9uLL);) {
            const float x = vertices[value_index++];
            const float y = vertices[value_index++];
            const float z = vertices[value_index++];

            this->min_x = std::min(this->min_x, x);
            this->min_y = std::min(this->min_y, y);
            this->min_z = std::min(this->min_z, z);

            this->max_x = std::max(this->max_x, x);
            this->max_y = std::max(this->max_y, y);
            this->max_z = std::max(this->max_z, z);
        }

        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Extent: [%.2f, %.2f, %.2f] x [%.2f, %.2f, %.2f]",
            this->min_x, this->min_y, this->min_z, this->max_x, this->max_y, this->max_z);
    }

    return true;
}

bool STLDataSource::get_extent_callback(core::Call& caller) {
    // Get mesh call
    auto& call = dynamic_cast<core::AbstractGetData3DCall&>(caller);

    get_input();

    call.SetExtent(
        static_cast<unsigned int>(1), this->min_x, this->min_y, this->min_z, this->max_x, this->max_y, this->max_z);

    return true;
}

bool STLDataSource::get_mesh_data_callback(core::Call& caller) {
    // Get mesh call
    auto& call = dynamic_cast<geocalls_gl::CallTriMeshDataGL&>(caller);

    if (call.DataHash() != static_cast<SIZE_T>(hash())) {
        // Read data if necessary
        if (this->filename_slot.IsDirty()) {
            if (!get_extent_callback(caller)) {
                return false;
            }
        }

        call.SetDataHash(static_cast<SIZE_T>(hash()));

        // Fill call
        this->mesh.SetVertexData(static_cast<unsigned int>(this->num_triangles * 3), this->vertex_buffer->data(),
            this->normal_buffer->data(), nullptr, nullptr, false);

        this->mesh.SetTriangleData(static_cast<unsigned int>(this->num_triangles), this->index_buffer->data(), false);

        call.SetObjects(1, &this->mesh);
    }

    return true;
}

void STLDataSource::release() {}

void STLDataSource::read(const std::string& filename) {
    std::ifstream ifs(filename, std::ofstream::in | std::ofstream::binary);

    if (ifs.good()) {
        // Read first non-white characters to identify an ASCII file
        std::string line;

        while (line.empty() && !ifs.eof()) {
            // Read and convert line to lower case
            std::getline(ifs, line);
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);

            line.erase(0, line.find_first_not_of("\f\n\r\t\v "));
        }

        // Read file
        if (line.substr(0, line.find_first_of("\f\n\r\t\v ")).compare("solid") == 0) {
            read_ascii(filename);
        } else {
            read_binary(filename);
        }
    } else {
        std::stringstream ss;
        ss << "STL file '" << filename << "' not found or inaccessible";

        throw std::runtime_error(ss.str());
    }
}

void STLDataSource::read_binary(const std::string& filename) {
    std::ifstream ifs(filename, std::ofstream::in | std::ofstream::binary);

    if (ifs.good()) {
        // Get number of triangles from header
        ifs.ignore(80 * sizeof(uint8_t));
        ifs.read(reinterpret_cast<char*>(&this->num_triangles), sizeof(uint32_t));

        // Sanity check for file size
        ifs.ignore(std::numeric_limits<std::streamsize>::max());
        std::streamsize file_size = ifs.gcount();

        if (file_size != this->num_triangles * 50uLL) {
            throw std::runtime_error("File size does not match the number of triangles.");
        }

        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "Number of triangles from binary STL file: %d", this->num_triangles);

        // Read data
        ifs.seekg(80 * sizeof(uint8_t) + sizeof(uint32_t));


        this->vertex_buffer->resize(9uLL * this->num_triangles);
        this->normal_buffer->resize(9uLL * this->num_triangles);

        for (std::size_t triangle_index = 0; triangle_index < static_cast<std::size_t>(this->num_triangles);
             ++triangle_index) {

            const auto data_index = 9uLL * triangle_index;

            ifs.read(reinterpret_cast<char*>(&(*this->normal_buffer)[data_index]), 3 * sizeof(float));
            ifs.read(reinterpret_cast<char*>(&(*this->vertex_buffer)[data_index]), 9 * sizeof(float));
            ifs.ignore(sizeof(uint16_t));

            std::memcpy(&(*this->normal_buffer)[data_index + 3 * sizeof(float)], &(*this->normal_buffer)[data_index],
                3 * sizeof(float));
            std::memcpy(&(*this->normal_buffer)[data_index + 6 * sizeof(float)], &(*this->normal_buffer)[data_index],
                3 * sizeof(float));
        }

        // Fill index buffer
        this->index_buffer->resize(3uLL * this->num_triangles);

        std::iota(this->index_buffer->begin(), this->index_buffer->end(), 0);
    } else {
        std::stringstream ss;
        ss << "STL file '" << filename << "' not found or inaccessible";

        throw std::runtime_error(ss.str());
    }
}

void STLDataSource::read_ascii(const std::string& filename) {
    std::ifstream ifs(filename, std::ofstream::in);

    if (ifs.good()) {
        // Get file size and estimate number of triangles
        ifs.ignore(std::numeric_limits<std::streamsize>::max());
        std::streamsize file_size = ifs.gcount();
        ifs.seekg(ifs.beg);

        const std::size_t estimated_num_triangles =
            static_cast<std::size_t>(1.5 * (static_cast<float>(file_size) / 256.0f));

        this->vertex_buffer->reserve(9 * estimated_num_triangles);
        this->normal_buffer->reserve(9 * estimated_num_triangles);

        // Parse file
        parser_states_t state = parser_states_t::EXPECT_SOLID;

        std::string line;
        std::string name;

        while (!ifs.eof()) {
            // Read and convert line to lower case
            std::getline(ifs, line);
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);

            line.erase(0, line.find_first_not_of("\f\n\r\t\v "));

            while (!line.empty()) {
                // Extract word from line
                const std::string word = line.substr(0, line.find_first_of("\f\n\r\t\v "));

                line.erase(0, word.size());
                line.erase(0, line.find_first_not_of("\f\n\r\t\v "));

                // Parse word
                switch (state) {
                case parser_states_t::EXPECT_SOLID:
                    if (word.compare("solid") == 0) {
                        state = parser_states_t::EXPECT_NAME;
                    } else {
                        throw std::runtime_error("Expected ASCII STL file to begin with keyword 'solid'");
                    }

                    break;
                case parser_states_t::EXPECT_NAME:
                    name = word;

                    state = parser_states_t::EXPECT_FACET_OR_ENDSOLID;

                    break;
                case parser_states_t::EXPECT_FACET_OR_ENDSOLID:
                    if (word.compare("facet") == 0) {
                        state = parser_states_t::EXPECT_NORMAL;
                    } else if (word.compare("endsolid") == 0) {
                        state = parser_states_t::EXPECT_ENDNAME;
                    } else {
                        throw std::runtime_error("Expected keyword 'facet' or 'endsolid' after solid name or "
                                                 "'endfacet' in ASCII STL file");
                    }

                    break;
                case parser_states_t::EXPECT_NORMAL:
                    if (word.compare("normal") == 0) {
                        state = parser_states_t::EXPECT_NORMAL_I;
                    } else {
                        throw std::runtime_error("Expected keyword 'normal' after keyword 'facet' in ASCII STL file");
                    }

                    break;
                case parser_states_t::EXPECT_NORMAL_I:
                    this->normal_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_NORMAL_J;

                    break;
                case parser_states_t::EXPECT_NORMAL_J:
                    this->normal_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_NORMAL_K;

                    break;
                case parser_states_t::EXPECT_NORMAL_K:
                    this->normal_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    // Duplicate normal for each vertex
                    for (int i = 0; i < 6; ++i) {
                        this->normal_buffer->push_back(this->normal_buffer->at(this->normal_buffer->size() - 3));
                    }

                    state = parser_states_t::EXPECT_OUTER;

                    break;
                case parser_states_t::EXPECT_OUTER:
                    if (word.compare("outer") == 0) {
                        state = parser_states_t::EXPECT_LOOP;
                    } else {
                        throw std::runtime_error("Expected keyword 'outer' after the normal values in ASCII STL file");
                    }

                    break;
                case parser_states_t::EXPECT_LOOP:
                    if (word.compare("loop") == 0) {
                        state = parser_states_t::EXPECT_VERTEX_1;
                    } else {
                        throw std::runtime_error("Expected keyword 'loop' after keyword 'outer' in ASCII STL file");
                    }

                    break;
                case parser_states_t::EXPECT_VERTEX_1:
                    if (word.compare("vertex") == 0) {
                        state = parser_states_t::EXPECT_VERTEX_1_X;
                    } else {
                        throw std::runtime_error("Expected keyword 'vertex' after keyword 'loop' in ASCII STL file");
                    }

                    break;
                case parser_states_t::EXPECT_VERTEX_1_X:
                    this->vertex_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_VERTEX_1_Y;

                    break;
                case parser_states_t::EXPECT_VERTEX_1_Y:
                    this->vertex_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_VERTEX_1_Z;

                    break;
                case parser_states_t::EXPECT_VERTEX_1_Z:
                    this->vertex_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_VERTEX_2;

                    break;
                case parser_states_t::EXPECT_VERTEX_2:
                    if (word.compare("vertex") == 0) {
                        state = parser_states_t::EXPECT_VERTEX_2_X;
                    } else {
                        throw std::runtime_error(
                            "Expected keyword 'vertex' after the first vertex values in ASCII STL file");
                    }

                    break;
                case parser_states_t::EXPECT_VERTEX_2_X:
                    this->vertex_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_VERTEX_2_Y;

                    break;
                case parser_states_t::EXPECT_VERTEX_2_Y:
                    this->vertex_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_VERTEX_2_Z;

                    break;
                case parser_states_t::EXPECT_VERTEX_2_Z:
                    this->vertex_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_VERTEX_3;

                    break;
                case parser_states_t::EXPECT_VERTEX_3:
                    if (word.compare("vertex") == 0) {
                        state = parser_states_t::EXPECT_VERTEX_3_X;
                    } else {
                        throw std::runtime_error(
                            "Expected keyword 'vertex' after the second vertex values in ASCII STL file");
                    }

                    break;
                case parser_states_t::EXPECT_VERTEX_3_X:
                    this->vertex_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_VERTEX_3_Y;

                    break;
                case parser_states_t::EXPECT_VERTEX_3_Y:
                    this->vertex_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_VERTEX_3_Z;

                    break;
                case parser_states_t::EXPECT_VERTEX_3_Z:
                    this->vertex_buffer->push_back(static_cast<float>(atof(word.c_str())));

                    state = parser_states_t::EXPECT_ENDLOOP;

                    break;
                case parser_states_t::EXPECT_ENDLOOP:
                    if (word.compare("endloop") == 0) {
                        state = parser_states_t::EXPECT_ENDFACET;
                    } else {
                        throw std::runtime_error(
                            "Expected keyword 'endloop' after the third vertex values in ASCII STL file");
                    }

                    break;
                case parser_states_t::EXPECT_ENDFACET:
                    if (word.compare("endfacet") == 0) {
                        state = parser_states_t::EXPECT_FACET_OR_ENDSOLID;
                    } else {
                        throw std::runtime_error(
                            "Expected keyword 'endfacet' after keyword 'endloop' in ASCII STL file");
                    }

                    break;
                case parser_states_t::EXPECT_ENDNAME:
                    if (word.compare(name) == 0) {
                        state = parser_states_t::EXPECT_EOF;
                    } else {
                        throw std::runtime_error("Expected same solid name after keyword 'endsolid' in ASCII STL file");
                    }

                    break;
                case parser_states_t::EXPECT_EOF:
                default:
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "Found more text after keyword 'endsolid' in ASCII STL file. %s",
                        "Maybe more than one object is stored in the file. This is not supported by this "
                        "reader.");
                }
            }
        }

        this->num_triangles = static_cast<uint32_t>(this->vertex_buffer->size() / 9);
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "Number of triangles from ASCII STL file: %d", this->num_triangles);

        // Fill index buffer
        this->index_buffer->resize(3uLL * this->num_triangles);

        std::iota(this->index_buffer->begin(), this->index_buffer->end(), 0);
    }
}

uint32_t STLDataSource::hash() const {
    if (this->vertex_buffer->empty()) {
        return 0;
    }

    return core::utility::DataHash(
        // Header
        this->vertex_buffer->size(),
        // First vertex
        *(this->vertex_buffer->cbegin() + 0), *(this->vertex_buffer->cbegin() + 1),
        *(this->vertex_buffer->cbegin() + 2),
        // Last vertex
        *(this->vertex_buffer->cend() - 1), *(this->vertex_buffer->cend() - 2), *(this->vertex_buffer->cend() - 3),
        // First normal
        *(this->normal_buffer->cbegin() + 0), *(this->normal_buffer->cbegin() + 1),
        *(this->normal_buffer->cbegin() + 2),
        // Last normal
        *(this->normal_buffer->cend() - 1), *(this->normal_buffer->cend() - 2), *(this->normal_buffer->cend() - 3));
}
} // namespace io
} // namespace datatools_gl
} // namespace megamol
