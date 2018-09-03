/*
 * PLYDataSource.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <vector>
#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AnimDataModule.h"
#include "tinyply.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Log.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace io {


/**
 * Data source module for .PLY files.
 */
class PLYDataSource : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "PLYDataSource"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source module for .PLY files."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /**
     * Constructor.
     */
    PLYDataSource(void);

    /**
     * Destructor.
     */
    virtual ~PLYDataSource(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * Reads the data from an open instream.
     *
     * @return True on success, false otherwise.
     */
    bool assertData(void);

    /**
     * Checks whether one the input parameters is dirty and resets the dirtyness state
     *
     * @return True if one of the parameters was dirty, false otherwise.
     */
    bool checkParameterDirtyness(void);

    /**
     * Clears all allocated fields.
     */
    void clearAllFields(void);

    /**
     * Callback that is called when the filename is changed.
     *
     * @param slot The slot containing the file path.
     * @return True on success, false otherwise.
     */
    bool filenameChanged(core::param::ParamSlot& slot);

    /**
     * Updates the currently read values according to the new selections.
     * 
     * @param slot The slot containing the file path.
     * @return True on success, false otherwise.
     */
    bool fileUpdate(core::param::ParamSlot& slot);

    /**
     * Callback setting the data of the read spheres.
     *
     * @param caller The calling call.
     * @return True on success, false otherwise.
     */
    bool getSphereDataCallback(core::Call& caller);

    /**
     * Callback setting the extent of the read sphere data.
     *
     * @param caller The calling call.
     * @return True on success, false otherwise.
     */
    bool getSphereExtentCallback(core::Call& caller);

    /**
     * Callback setting the data of the read mesh.
     *
     * @param caller The calling call.
     * @return True on success, false otherwise.
     */
    bool getMeshDataCallback(core::Call& caller);

    /**
     * Callback setting the extent of the read mesh data.
     *
     * @param caller The calling call.
     * @return True on success, false otherwise.
     */
    bool getMeshExtentCallback(core::Call& caller);

    /**
     * Resets the dirtyness of all parameters
     */
    void resetParameterDirtyness(void);

    /** Slot for the filepath of the .ply file */
    core::param::ParamSlot filename;

    /** Slot for the vertex element name */
    core::param::ParamSlot vertElemSlot;

    /** Slot for the face element name */
    core::param::ParamSlot faceElemSlot;

    /** Slot for the x property name */
    core::param::ParamSlot xPropSlot;

    /** Slot for the y property name */
    core::param::ParamSlot yPropSlot;

    /** Slot for the z property name */
    core::param::ParamSlot zPropSlot;

    /** Slot for the nx property name */
    core::param::ParamSlot nxPropSlot;

    /** Slot for the nx property name */
    core::param::ParamSlot nyPropSlot;

    /** Slot for the nz property name */
    core::param::ParamSlot nzPropSlot;

    /** Slot for the r property name */
    core::param::ParamSlot rPropSlot;

    /** Slot for the g property name */
    core::param::ParamSlot gPropSlot;

    /** Slot for the b property name */
    core::param::ParamSlot bPropSlot;

    /** Slot for the i property name */
    core::param::ParamSlot iPropSlot;

    /** Slot for the index property name */
    core::param::ParamSlot indexPropSlot;

    /** Slot for the uniform sphere radius */
    core::param::ParamSlot radiusSlot;

    /** Guessed and real names of the position properties */
    std::vector<std::string> guessedPos, selectedPos;

    /** Guessed and real names of the normal properties */
    std::vector<std::string> guessedNormal, selectedNormal;

    /** Guessed and real names of the color properties */
    std::vector<std::string> guessedColor, selectedColor;

    /** Guessed and real names of the index property */
    std::string guessedIndices, selectedIndices;

    /** Guessed and real names of the vertex property */
    std::string guessedVertices, selectedVertices;

    /** Guessed and real names of the face property */
    std::string guessedFaces, selectedFaces;

    /** Sizes in byte for each element of the read ply file */
    std::vector<uint64_t> elementSizes;

    /** Count of data points for each element */
    std::vector<uint64_t> elementCount;

	/** The names for each element */
	std::vector<std::string> elementNames;

    /** Sizes in byte for each property of the read ply file. The size is splitted into the sizes of each separate
     * element */
    std::vector<std::vector<uint64_t>> propertySizes;

    /** Signs of the property values. True if the value is signed, false if it is unsigned*/
    std::vector<std::vector<bool>> propertySigns;

    /** The strides of each property element */
    std::vector<std::vector<uint64_t>> propertyStrides;

    /** Flags showing whether a property is a list of values */
    std::vector<std::vector<bool>> listFlags;

    /** Size of the list headers, if present */
    std::vector<std::vector<uint64_t>> listSizes;

    /** Signs of the list header sizes, if present */
    std::vector<std::vector<bool>> listSigns;

    /** Slot offering the sphere data. */
    core::CalleeSlot getSphereData;

    /** Slot offering the mesh data. */
    core::CalleeSlot getMeshData;

    /** The input file stream. */
    std::ifstream instream;

    /** Struct for the different possible position types */
    struct pos_type {
        double* pos_double = nullptr;
        float* pos_float = nullptr;
    } posPointers;

    /** Struct for the different possible color types */
    struct col_type {
        unsigned char* col_uchar = nullptr;
        float* col_float = nullptr;
        double* col_double = nullptr;
    } colorPointers;

    /** Struct for the different possible normal types */
    struct normal_type {
        double* norm_double = nullptr;
        float* norm_float = nullptr;
    } normalPointers;

    /** Struct for the different possible face types */
    struct face_type {
        uint8_t* face_uchar = nullptr;
        uint16_t* face_u16 = nullptr;
        uint32_t* face_u32 = nullptr;
    } facePointers;

    /** Map for the element names to their indices*/
    std::map<std::string, std::pair<uint64_t, uint64_t>> elementIndexMap;

    /** Flag determining the file format of the read file */
    bool hasBinaryFormat;

    /** Flag determining the endianness if the read file has binary format */
    bool isLittleEndian;

    /** The bounding box of the vertices */
    vislib::math::Cuboid<float> boundingBox;

    /** The bounding box including sphere radii */
    vislib::math::Cuboid<float> sphereBoundingBox;

    /** The mesh data for the triangle data call */
    geocalls::CallTriMeshData::Mesh mesh;

    /** The number of vertices */
    size_t vertex_count;

    /** The number of faces */
    size_t face_count;

    /** The offset the first data point has from the start of the read file */
    size_t data_offset;

    /** The current data hash. */
    size_t data_hash;
};

} /* end namespace io */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED */
