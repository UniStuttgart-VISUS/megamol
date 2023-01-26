/*
 * STLDataSource.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_IO_STLDATASOURCE_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_STLDATASOURCE_H_INCLUDED
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "geometry_calls_gl/CallTriMeshDataGL.h"

#include <memory>
#include <string>
#include <vector>

namespace megamol {
namespace datatools_gl {
namespace io {
class STLDataSource : public core::Module {
public:
    /// <summary>
    /// Answer the name of this module
    /// </summary>
    /// <returns>Module name</returns>
    static const char* ClassName() {
        return "STLDataSource";
    }

    /// <summary>
    /// Answer a human-readable description
    /// </summary>
    /// <returns>Description</returns>
    static const char* Description() {
        return "Reader for triangle data in STL files";
    }

    /// <summary>
    /// Answer the availability of this module
    /// </summary>
    /// <returns>Availability</returns>
    static bool IsAvailable() {
        return true;
    }

    /// <summary>
    /// Constructor
    /// </summary>
    STLDataSource();

    /// <summary>
    /// Destructor
    /// </summary>
    ~STLDataSource() override;

protected:
    /// <summary>
    /// Create the module
    /// </summary>
    /// <returns>True on success; false otherwise</returns>
    bool create() override;

    /// <summary>
    /// Read in file and store data
    /// </summary>
    /// <returns>Success</returns>
    bool get_input();

    /// <summary>
    /// Callback function for requesting information
    /// </summary>
    /// <param name="caller">Call for this request</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool get_extent_callback(core::Call& caller);

    /// <summary>
    /// Callback function for requesting data
    /// </summary>
    /// <param name="caller">Call for this request</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool get_mesh_data_callback(core::Call& caller);

    /// <summary>
    /// Release the module
    /// </summary>
    void release() override;

private:
    /// <summary>
    /// States for the ascii parser
    /// </summary>
    enum class parser_states_t {
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
    void read(const std::string& filename);

    /// <summary>
    /// Read a binary file
    /// </summary>
    /// <param name="filename">File name of the STL file</param>
    void read_binary(const std::string& filename);

    /// <summary>
    /// Read a textual file
    /// </summary>
    /// <param name="filename">File name of the STL file</param>
    void read_ascii(const std::string& filename);

    /// File name
    core::param::ParamSlot filename_slot;

protected:
    /// <summary>
    /// Calculate the data hash
    /// </summary>
    /// <returns>Data hash</returns>
    uint32_t hash() const;

    /// Output
    core::CalleeSlot mesh_output_slot;

    /// Mesh data
    geocalls_gl::CallTriMeshDataGL::Mesh mesh;

    /// Buffers to store vertices and normals, and indices
    uint32_t num_triangles;

    float min_x, min_y, min_z, max_x, max_y, max_z;

    std::shared_ptr<std::vector<float>> vertex_buffer, normal_buffer;
    std::shared_ptr<std::vector<unsigned int>> index_buffer;
};
} // namespace io
} // namespace datatools_gl
} // namespace megamol

#endif // !MEGAMOL_DATATOOLS_IO_STLDATASOURCE_H_INCLUDED
