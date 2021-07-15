/*
 * AbstractSTLDataSource.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "mmcore/utility/DataHash.h"

#include <memory>
#include <string>
#include <vector>

namespace megamol ::stdplugin::datatools::io {
template<typename OutputCall>
class AbstractSTLDataSource : public core::Module {
public:
    /// <summary>
    /// Constructor
    /// </summary>
    AbstractSTLDataSource()
            : filename_slot("STL file", "The name of to the STL file to load")
            , mesh_output_slot("mesh_data", "Slot to request mesh data") {
        // Create file name textbox
        this->filename_slot << new core::param::FilePathParam("");
        Module::MakeSlotAvailable(&this->filename_slot);

        // Create output slot for triangle mesh data
        this->mesh_output_slot.SetCallback(
            OutputCall::ClassName(), OutputCall::FunctionName(1), &AbstractSTLDataSource::get_extent_callback);
        this->mesh_output_slot.SetCallback(
            OutputCall::ClassName(), OutputCall::FunctionName(0), &AbstractSTLDataSource::get_mesh_data_callback);

        Module::MakeSlotAvailable(&this->mesh_output_slot);
    }

    /// <summary>
    /// Destructor
    /// </summary>
    virtual ~AbstractSTLDataSource() {
        this->Release();
    }

protected:
    /// <summary>
    /// Create the module
    /// </summary>
    /// <returns>True on success; false otherwise</returns>
    virtual bool create() override {
        return true;
    }

    /// <summary>
    /// Callback function for requesting information
    /// </summary>
    /// <param name="caller">Call for this request</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool get_extent_callback(core::Call& caller) = 0;

    /// <summary>
    /// Callback function for requesting data
    /// </summary>
    /// <param name="caller">Call for this request</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool get_mesh_data_callback(core::Call& caller) = 0;

    /// <summary>
    /// Release the module
    /// </summary>
    virtual void release() override {}

//private:
    /// <summary>
    /// States for the ascii parser
    /// </summary>
    enum parser_states_t {
        EXPECT_SOLID,
        EXPECT_NAME,
        EXPECT_FACET_OR_ENDSOLID,
        EXPECT_NORMAL,
        EXPECT_NORMAL_I,
        EXPECT_NORMAL_J,
        EXPECT_NORMAL_K,
        EXPECT_OUTER,
        EXPECT_LOOP,
        EXPECT_VERTEX_1,
        EXPECT_VERTEX_1_X,
        EXPECT_VERTEX_1_Y,
        EXPECT_VERTEX_1_Z,
        EXPECT_VERTEX_2,
        EXPECT_VERTEX_2_X,
        EXPECT_VERTEX_2_Y,
        EXPECT_VERTEX_2_Z,
        EXPECT_VERTEX_3,
        EXPECT_VERTEX_3_X,
        EXPECT_VERTEX_3_Y,
        EXPECT_VERTEX_3_Z,
        EXPECT_ENDLOOP,
        EXPECT_ENDFACET,
        EXPECT_ENDNAME,
        EXPECT_EOF
    };

    /// <summary>
    /// Read an STL file
    /// </summary>
    /// <param name="filename">File name of the STL file</param>
    void read(const std::string& filename) {
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

    /// <summary>
    /// Read a binary file
    /// </summary>
    /// <param name="filename">File name of the STL file</param>
    void read_binary(const std::string& filename) {
        std::ifstream ifs(filename, std::ofstream::in | std::ofstream::binary);

        if (ifs.good()) {
            // Get number of triangles from header
            ifs.ignore(80 * sizeof(uint8_t));
            ifs.read(reinterpret_cast<char*>(&this->num_triangles), sizeof(uint32_t));

            // Sanity check for file size
            ifs.ignore(std::numeric_limits<std::streamsize>::max());
            std::streamsize file_size = ifs.gcount();

            if (file_size != this->num_triangles * 50) {
                throw std::runtime_error("File size does not match the number of triangles.");
            }

            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "Number of triangles from binary STL file: %d", this->num_triangles);

            // Read data
            ifs.seekg(80 * sizeof(uint8_t) + sizeof(uint32_t));

            const std::size_t header_block_size = sizeof(uint32_t);
            const std::size_t data_block_size = 9 * this->num_triangles * sizeof(float);

            const std::size_t data_offset_1 = 2 * header_block_size;
            const std::size_t data_offset_2 = 2 * header_block_size + data_block_size;

            this->vertex_normal_buffer.resize(2 * data_block_size + 2 * header_block_size);

            reinterpret_cast<uint32_t&>(this->vertex_normal_buffer[0 * header_block_size]) =
                static_cast<uint32_t>(data_offset_1);
            reinterpret_cast<uint32_t&>(this->vertex_normal_buffer[1 * header_block_size]) =
                static_cast<uint32_t>(data_offset_2);

            std::size_t vertex_position = data_offset_1;
            std::size_t normal_position = data_offset_2;

            for (std::size_t triangle_index = 0; triangle_index < static_cast<std::size_t>(this->num_triangles);
                 ++triangle_index) {
                ifs.read(reinterpret_cast<char*>(&this->vertex_normal_buffer[normal_position]), 3 * sizeof(float));
                ifs.read(reinterpret_cast<char*>(&this->vertex_normal_buffer[vertex_position]), 9 * sizeof(float));
                ifs.ignore(sizeof(uint16_t));

                std::memcpy(&this->vertex_normal_buffer[normal_position + 3 * sizeof(float)],
                    &this->vertex_normal_buffer[normal_position], 3 * sizeof(float));
                std::memcpy(&this->vertex_normal_buffer[normal_position + 6 * sizeof(float)],
                    &this->vertex_normal_buffer[normal_position], 3 * sizeof(float));

                vertex_position += 9 * sizeof(float);
                normal_position += 9 * sizeof(float);
            }

            // Fill index buffer
            this->index_buffer.resize(this->num_triangles * 3);

            std::iota(this->index_buffer.begin(), this->index_buffer.end(), 0);
        } else {
            std::stringstream ss;
            ss << "STL file '" << filename << "' not found or inaccessible";

            throw std::runtime_error(ss.str());
        }
    }

    /// <summary>
    /// Read a textual file
    /// </summary>
    /// <param name="filename">File name of the STL file</param>
    void read_ascii(const std::string& filename) {
        std::ifstream ifs(filename, std::ofstream::in);

        if (ifs.good()) {
            // Get file size and estimate number of triangles
            ifs.ignore(std::numeric_limits<std::streamsize>::max());
            std::streamsize file_size = ifs.gcount();
            ifs.seekg(ifs.beg);

            const std::size_t estimated_num_triangles =
                static_cast<std::size_t>(1.5f * (static_cast<float>(file_size) / 256.0f));

            std::vector<float> vertices, normals;
            vertices.reserve(9 * estimated_num_triangles);
            normals.reserve(3 * estimated_num_triangles);

            // Parse file
            parser_states_t state = EXPECT_SOLID;

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
                    case EXPECT_SOLID:
                        if (word.compare("solid") == 0) {
                            state = EXPECT_NAME;
                        } else {
                            throw std::runtime_error("Expected ASCII STL file to begin with keyword 'solid'");
                        }

                        break;
                    case EXPECT_NAME:
                        name = word;

                        state = EXPECT_FACET_OR_ENDSOLID;

                        break;
                    case EXPECT_FACET_OR_ENDSOLID:
                        if (word.compare("facet") == 0) {
                            state = EXPECT_NORMAL;
                        } else if (word.compare("endsolid") == 0) {
                            state = EXPECT_ENDNAME;
                        } else {
                            throw std::runtime_error("Expected keyword 'facet' or 'endsolid' after solid "
                                                     "name or 'endfacet' in ASCII STL file");
                        }

                        break;
                    case EXPECT_NORMAL:
                        if (word.compare("normal") == 0) {
                            state = EXPECT_NORMAL_I;
                        } else {
                            throw std::runtime_error(
                                "Expected keyword 'normal' after keyword 'facet' in ASCII STL file");
                        }

                        break;
                    case EXPECT_NORMAL_I:
                        normals.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_NORMAL_J;

                        break;
                    case EXPECT_NORMAL_J:
                        normals.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_NORMAL_K;

                        break;
                    case EXPECT_NORMAL_K:
                        normals.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_OUTER;

                        break;
                    case EXPECT_OUTER:
                        if (word.compare("outer") == 0) {
                            state = EXPECT_LOOP;
                        } else {
                            throw std::runtime_error(
                                "Expected keyword 'outer' after the normal values in ASCII STL file");
                        }

                        break;
                    case EXPECT_LOOP:
                        if (word.compare("loop") == 0) {
                            state = EXPECT_VERTEX_1;
                        } else {
                            throw std::runtime_error("Expected keyword 'loop' after keyword 'outer' in ASCII STL file");
                        }

                        break;
                    case EXPECT_VERTEX_1:
                        if (word.compare("vertex") == 0) {
                            state = EXPECT_VERTEX_1_X;
                        } else {
                            throw std::runtime_error(
                                "Expected keyword 'vertex' after keyword 'loop' in ASCII STL file");
                        }

                        break;
                    case EXPECT_VERTEX_1_X:
                        vertices.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_VERTEX_1_Y;

                        break;
                    case EXPECT_VERTEX_1_Y:
                        vertices.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_VERTEX_1_Z;

                        break;
                    case EXPECT_VERTEX_1_Z:
                        vertices.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_VERTEX_2;

                        break;
                    case EXPECT_VERTEX_2:
                        if (word.compare("vertex") == 0) {
                            state = EXPECT_VERTEX_2_X;
                        } else {
                            throw std::runtime_error("Expected keyword 'vertex' after the first vertex "
                                                     "values in ASCII STL file");
                        }

                        break;
                    case EXPECT_VERTEX_2_X:
                        vertices.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_VERTEX_2_Y;

                        break;
                    case EXPECT_VERTEX_2_Y:
                        vertices.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_VERTEX_2_Z;

                        break;
                    case EXPECT_VERTEX_2_Z:
                        vertices.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_VERTEX_3;

                        break;
                    case EXPECT_VERTEX_3:
                        if (word.compare("vertex") == 0) {
                            state = EXPECT_VERTEX_3_X;
                        } else {
                            throw std::runtime_error("Expected keyword 'vertex' after the second vertex "
                                                     "values in ASCII STL file");
                        }

                        break;
                    case EXPECT_VERTEX_3_X:
                        vertices.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_VERTEX_3_Y;

                        break;
                    case EXPECT_VERTEX_3_Y:
                        vertices.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_VERTEX_3_Z;

                        break;
                    case EXPECT_VERTEX_3_Z:
                        vertices.push_back(static_cast<float>(atof(word.c_str())));

                        state = EXPECT_ENDLOOP;

                        break;
                    case EXPECT_ENDLOOP:
                        if (word.compare("endloop") == 0) {
                            state = EXPECT_ENDFACET;
                        } else {
                            throw std::runtime_error("Expected keyword 'endloop' after the third vertex "
                                                     "values in ASCII STL file");
                        }

                        break;
                    case EXPECT_ENDFACET:
                        if (word.compare("endfacet") == 0) {
                            state = EXPECT_FACET_OR_ENDSOLID;
                        } else {
                            throw std::runtime_error(
                                "Expected keyword 'endfacet' after keyword 'endloop' in ASCII STL file");
                        }

                        break;
                    case EXPECT_ENDNAME:
                        if (word.compare(name) == 0) {
                            state = EXPECT_EOF;
                        } else {
                            throw std::runtime_error(
                                "Expected same solid name after keyword 'endsolid' in ASCII STL file");
                        }

                        break;
                    case EXPECT_EOF:
                    default:
                        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                            "Found more text after keyword 'endsolid' in ASCII STL file. %s",
                            "Maybe more than one object is stored in the file. This is not supported by "
                            "this reader.");
                    }
                }
            }

            this->num_triangles = static_cast<uint32_t>(vertices.size() / 9);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "Number of triangles from ASCII STL file: %d", this->num_triangles);

            // Fill buffer
            this->vertex_normal_buffer.resize(2 * sizeof(uint32_t) + 18 * this->num_triangles * sizeof(float));

            reinterpret_cast<uint32_t*>(this->vertex_normal_buffer.data())[0] = 2 * sizeof(uint32_t);
            reinterpret_cast<uint32_t*>(this->vertex_normal_buffer.data())[1] =
                2 * sizeof(uint32_t) + 9 * this->num_triangles * sizeof(float);

            std::memcpy(&this->vertex_normal_buffer[2 * sizeof(uint32_t)], vertices.data(),
                9 * this->num_triangles * sizeof(float));

            const std::size_t offset = 2 * sizeof(uint32_t) + 9 * this->num_triangles * sizeof(float);
            std::size_t position = 0;

            for (std::size_t triangle_index = 0; triangle_index < this->num_triangles; ++triangle_index) {
                reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] =
                    normals[triangle_index * 3 + 0];
                reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] =
                    normals[triangle_index * 3 + 1];
                reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] =
                    normals[triangle_index * 3 + 2];

                reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] =
                    normals[triangle_index * 3 + 0];
                reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] =
                    normals[triangle_index * 3 + 1];
                reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] =
                    normals[triangle_index * 3 + 2];

                reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] =
                    normals[triangle_index * 3 + 0];
                reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] =
                    normals[triangle_index * 3 + 1];
                reinterpret_cast<float*>(&this->vertex_normal_buffer[offset])[position++] =
                    normals[triangle_index * 3 + 2];
            }

            // Fill index buffer
            this->index_buffer.resize(this->num_triangles * 3);

            std::iota(this->index_buffer.begin(), this->index_buffer.end(), 0);
        }
    }

    /// <summary>
    /// Calculate the data hash
    /// </summary>
    /// <returns>Data hash</returns>
    uint32_t hash() const {
        if (this->vertex_normal_buffer.empty()) {
            return 0;
        }

        const uint32_t first_dataset =
            reinterpret_cast<const uint32_t&>(this->vertex_normal_buffer[0 * sizeof(uint32_t)]);
        const uint32_t second_dataset =
            reinterpret_cast<const uint32_t&>(this->vertex_normal_buffer[1 * sizeof(uint32_t)]);

        return core::utility::DataHash(
            // Header
            first_dataset, second_dataset,
            // First vertex
            reinterpret_cast<const float&>(this->vertex_normal_buffer[first_dataset + 0 * sizeof(float)]),
            reinterpret_cast<const float&>(this->vertex_normal_buffer[first_dataset + 1 * sizeof(float)]),
            reinterpret_cast<const float&>(this->vertex_normal_buffer[first_dataset + 2 * sizeof(float)]),
            // Last vertex
            reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset - 3 * sizeof(float)]),
            reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset - 2 * sizeof(float)]),
            reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset - 1 * sizeof(float)]),
            // First normal
            reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset + 0 * sizeof(float)]),
            reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset + 1 * sizeof(float)]),
            reinterpret_cast<const float&>(this->vertex_normal_buffer[second_dataset + 2 * sizeof(float)]),
            // Last normal
            reinterpret_cast<const float&>(
                this->vertex_normal_buffer[2 * second_dataset - first_dataset - 3 * sizeof(float)]),
            reinterpret_cast<const float&>(
                this->vertex_normal_buffer[2 * second_dataset - first_dataset - 2 * sizeof(float)]),
            reinterpret_cast<const float&>(
                this->vertex_normal_buffer[2 * second_dataset - first_dataset - 1 * sizeof(float)]));
    }

    /// File name
    core::param::ParamSlot filename_slot;

    /// Output
    core::CalleeSlot mesh_output_slot;

    /// Mesh data
    //geocalls::CallTriMeshData::Mesh mesh;

    /// Buffers to store vertices and normals, and indices
    uint32_t num_triangles;

    float min_x, min_y, min_z, max_x, max_y, max_z;

    std::vector<uint8_t> vertex_normal_buffer;
    std::vector<unsigned int> index_buffer;
};
} // namespace megamol::stdplugin::datatools::io
