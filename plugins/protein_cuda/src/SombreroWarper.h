/*
 * SombreroWarper.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "SombreroKernels.cuh"
#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/TunnelResidueDataCall.h"

#include <set>

namespace megamol {
namespace protein_cuda {

/**
 * Class for the warping of a given mesh into a sombrero shape
 */
class SombreroWarper : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "SombreroWarper";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module that warps a given mesh to resemble the shape of a sombrero.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    SombreroWarper(void);

    /** Dtor. */
    virtual ~SombreroWarper(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'release'.
     */
    virtual void release(void);

    /**
     * Call for get data.
     */
    bool getData(megamol::core::Call& call);

    /**
     * Call for get extent.
     */
    bool getExtent(megamol::core::Call& call);

private:
    /**
     * structure representing a triangle
     */
    struct Triangle {
        uint v1; // first vertex index
        uint v2; // second vertex index
        uint v3; // third vertex index

        /** Ctor. */
        Triangle() {
            v1 = 0;
            v2 = 0;
            v3 = 0;
        }

        /** Ctor */
        Triangle(uint v1, uint v2, uint v3) {
            this->v1 = v1;
            this->v2 = v2;
            this->v3 = v3;
            // sort the vertices ascending
            if (this->v1 > this->v3)
                std::swap(this->v1, this->v3);
            if (this->v1 > this->v2)
                std::swap(this->v1, this->v2);
            if (this->v2 > this->v3)
                std::swap(this->v2, this->v3);
        }

        /** equality operator overload */
        inline bool operator==(Triangle& o) {
            return this->v1 == o.v1 && this->v2 == o.v2 && this->v3 == o.v3;
        }
    };

    /**
     * Checks for dirty parameters and sets the dirty flag, if needed
     */
    void checkParameters(void);

    /**
     * Copies the mesh data into local buffers to make modification possible.
     *
     * @param ctmd The incoming call with the source data.
     * @return True on success, false otherwise.
     */
    bool copyMeshData(megamol::geocalls_gl::CallTriMeshDataGL& ctmd);

    /**
     * Fills holes in the generated mesh that were detected beforehand by findSombreroBorder(void)
     *
     * @return True on success, false otherwise.
     */
    bool fillMeshHoles(void);

    /**
     * Searches for the outer vertices of the sombrero brim.
     *
     * @return True on success, false otherwise.
     */
    bool findSombreroBorder(void);

    /**
     * Warps the mesh to resemble a sombrero.
     *
     * @tunnelCall Call containing the tunnel data.
     * @return True on success, false otherwise.
     */
    bool warpMesh(protein_calls::TunnelResidueDataCall& tunnelCall);

    /**
     * Performs the vertex level lifting necessary for the sombrero computation
     *
     * @return True on success, false otherwise.
     */
    bool liftVertices(void);

    /**
     * Recomputes the distances of the vertices after vertex insertion.
     *
     * @return True on success, false otherwise
     */
    bool recomputeVertexDistances(void);

    /**
     * Recomputes the normals of the vertices.
     *
     * @return True on success, false otherwise
     */
    bool recomputeVertexNormals(protein_calls::TunnelResidueDataCall& tunnelCall);

    /**
     * Fixes the broken parts of the mesh
     *
     * @param maxDist The maximum allowed distance from the average position of the neighbors
     * @return True on success, false otherwise
     */
    bool fixBrokenMeshParts(float maxDist);

    /**
     * Computes the angles for each vertex using an adapted method by rahi and sharp
     *
     * @param The call containing the tunnel data.
     * @return True on success, false otherwise.
     */
    bool computeVertexAngles(protein_calls::TunnelResidueDataCall& tunnelCall);

    /**
     * Reconstructs the edge search structures based on the edges stored in the edge fields.
     * It is assumed, that the offset fields already have the correct size.
     *
     * @param index The index of the mesh the search structure should be updated
     * @param vertex_cnt The new vertex count of the mesh
     */
    void reconstructEdgeSearchStructures(uint index, uint vertex_cnt);

    /**
     * Computes the height (the y-coordinate) of each vertex.
     *
     * @param bsVertex The index of the binding site vertex
     * @return True on success, false otherwise.
     */
    bool computeHeightPerVertex(uint bsVertex);

    /**
     * Computes the the x- and z-coordinates of each vertex.
     * To do this, the height already has to be computed.
     * In flat mode, this method also assigns the new height values
     *
     * @return True on success, false otherwise.
     */
    bool computeXZCoordinatePerVertex(protein_calls::TunnelResidueDataCall& tunnelCall);

    /**
     * Divides the mesh in the brim and crown parts so that the lighting looks correct.
     *
     * @return True on success, false otherwise.
     */
    bool divideMeshForOutput(void);

    /** The lastly received data hash */
    SIZE_T lastDataHash;

    /** The offset to the lastly received hash */
    SIZE_T hashOffset;

    /** Slot for the mesh input. */
    core::CallerSlot meshInSlot;

    /** Slot for the tunnel input */
    core::CallerSlot tunnelInSlot;

    /** Slot for the ouput of the cut mesh */
    core::CalleeSlot warpedMeshOutSlot;

    /** Parameter for the minimal brim level */
    core::param::ParamSlot minBrimLevelParam;

    /** Parameter fo the maximum brim level */
    core::param::ParamSlot maxBrimLevelParam;

    /** Parameter for the maximum target distance after vertex lifting  */
    core::param::ParamSlot liftingTargetDistance;

    /** Parameter for the maximum allowed distance before lifting*/
    core::param::ParamSlot maxAllowedLiftingDistance;

    /** Parameter enabling the flattening of the sombrero */
    core::param::ParamSlot flatteningParam;

    /** Weight of the southern border */
    core::param::ParamSlot southBorderWeightParam;

    /** Height factor for the souther border */
    core::param::ParamSlot southBorderHeightFactor;

    /** Param switch for normal inversion */
    core::param::ParamSlot invertNormalParam;

    /** Param switch for the mesh fixing */
    core::param::ParamSlot fixMeshParam;

    /** The maximum allowed distance before the mesh is fixed */
    core::param::ParamSlot meshFixDistanceParam;

    /** New radius computed param */
    core::param::ParamSlot radiusSelectionSlot;

    /** Scaling parameter for the brim radius */
    core::param::ParamSlot brimScalingParam;

    /** Scaling parameter for the sombrero radius */
    core::param::ParamSlot radiusScalingParam;

    /** Scaling parameter for the length of the sombrero */
    core::param::ParamSlot lengthScalingParam;

    /** Vector containing the modified mesh data */
    std::vector<geocalls_gl::CallTriMeshDataGL::Mesh> meshVector;

    /** Vector containing the cut modified mesh data */
    std::vector<geocalls_gl::CallTriMeshDataGL::Mesh> outMeshVector;

    /** The vertex positions of the mesh */
    std::vector<std::vector<float>> vertices;

    /** The vertex normals of the mesh */
    std::vector<std::vector<float>> normals;

    /** The vertex normals of the crown */
    std::vector<std::vector<float>> crownNormals;

    /** The vertex normals of the brim */
    std::vector<std::vector<float>> brimNormals;

    /** The vertex colors of the mesh */
    std::vector<std::vector<unsigned char>> colors;

    /** The atom indices per vertex for the mesh */
    std::vector<std::vector<uint>> atomIndexAttachment;

    /** The vertex levels of the mesh */
    std::vector<std::vector<uint>> vertexLevelAttachment;

    /** The distance of each vertex to the binding site */
    std::vector<std::vector<uint>> bsDistanceAttachment;

    /** The newly computed binding site distances after vertex lifting */
    std::vector<std::vector<uint>> newBsDistances;

    /** The faces of the mesh */
    std::vector<std::vector<uint>> faces;

    /** The faces of the brim */
    std::vector<std::vector<uint>> brimFaces;

    /** The faces of the crown */
    std::vector<std::vector<uint>> crownFaces;

    /** The face edges in forward order */
    std::vector<std::vector<std::pair<uint, uint>>> edgesForward;

    /** The face edges in reverse order */
    std::vector<std::vector<std::pair<uint, uint>>> edgesReverse;

    /** The offset arrays for both forward and backward sorted edges (first: forward, second: backward) */
    std::vector<std::vector<std::pair<uint, uint>>> vertexEdgeOffsets;

    /** Set containing the indices of the border vertices */
    std::vector<std::set<uint>> borderVertices;

    /** Sets containing the indices of all different cut vertices */
    std::vector<std::vector<std::set<uint>>> cutVertices;

    /** vertex angles computed by rahi and sharp */
    std::vector<std::vector<float>> rahiAngles;

    /** Flags for each vertex if they belong to the brim */
    std::vector<std::vector<bool>> brimFlags;

    /** Indices of all vertices belonging to the brim */
    std::vector<std::vector<uint>> brimIndices;

    /** Unique pointer to the CUDA kernels */
    std::unique_ptr<SombreroKernels> cuda_kernels;

    /** The length of the sombrero */
    std::vector<float> sombreroLength;

    /** The width of the brim */
    std::vector<float> brimWidth;

    /** The inner radius of the sombrero ellipsoid */
    std::vector<float> sombreroRadius;

    /** */
    std::vector<float> sombreroRadiusNew;

    /** The bounding box of the sombrero */
    vislib::math::Cuboid<float> boundingBox;

    /** Flag set when a parameter is dirty */
    bool dirtyFlag;
};

} // namespace protein_cuda
} /* end namespace megamol */
