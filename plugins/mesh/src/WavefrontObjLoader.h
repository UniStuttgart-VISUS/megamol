/*
 * WavefrontObjLoader.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef WAVEFRONT_OBJ_LOADER_H_INCLUDED
#define WAVEFRONT_OBJ_LOADER_H_INCLUDED

#include "mesh/MeshCalls.h"
#include "mesh/mesh.h"
#include "mesh/MeshDataAccessCollection.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "tiny_obj_loader.h"

namespace megamol {
namespace mesh {

class MESH_API WavefrontObjLoader : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "WavefrontObjLoader"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source for simply loading a wavefront obj file from disk"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    WavefrontObjLoader();
    ~WavefrontObjLoader();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    bool getMetaDataCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    void release();

private:

    struct TinyObjModel 
    {
        tinyobj::attrib_t                attrib;
        std::vector<tinyobj::shape_t>    shapes;
        std::vector<tinyobj::material_t> materials;
    };

    /**
     * Representation of obj model as loaded by tinyobjloader
     */
    std::shared_ptr<TinyObjModel> m_obj_model;

    /**
    // * Internal storage for unpacked positions, i.e one position per vertex, three vertices per triangle
    // */
    //std::vector<float> m_positions;

    ///**
    // * Internal storage for unpacked normals, i.e one normal per vertex, three vertices per triangle
    // */
    //std::vector<float> m_normals;

    ///**
    // * Internal storage for unpacked texcoords, i.e one texcoord per vertex, three vertices per triangle
    // */
    //std::vector<float> m_texcoords;

    /**
     * 
     */
    std::vector<std::vector<unsigned int>> m_indices;

    /**
     * Shareable access to the internally stored mesh data from loaded obj file.
     */
    std::shared_ptr<MeshDataAccessCollection> m_mesh_data_access;

    /**
     * Meta data for communicating data updates, as well as data size
     */
    core::Spatial3DMetaData m_meta_data;

    /** The gltf file name */
    core::param::ParamSlot m_filename_slot;

    /** The slot for requesting data */
    megamol::core::CalleeSlot m_getData_slot;

    // TODO slot for chaining
};

} // namespace mesh
} // namespace megamol


#endif // !#ifndef WAVEFRONT_OBJ_LOADER_H_INCLUDED
