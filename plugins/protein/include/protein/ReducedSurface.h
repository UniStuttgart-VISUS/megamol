/*
 * ReducedSurface.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_REDUCEDSURFACE_H_INCLUDED
#define MEGAMOL_REDUCEDSURFACE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "protein_calls/MolecularDataCall.h"
#include "vislib/math/Quaternion.h"
#include <algorithm>
#include <list>
#include <set>
#include <vector>

namespace megamol::protein {

/**
 * Molecular Surface Renderer class.
 * Computes and renders the solvent excluded (Connolly) surface.
 */
class ReducedSurface {
public:
    class RSVertex;
    class RSEdge;
    class RSFace;

    /**
     * Reduced Surface (RS) Vertex class.
     * Stores the index of the vertex position, a pointer-list of edges that
     * the vertex belongs to and if it is buried inside the molecule.
     * An empty edge list of a non-burried vertex indicates a free RS-vertex.
     */
    class RSVertex {
    public:
        RSVertex(vislib::math::Vector<float, 3> pos, float rad, unsigned int atomIdx);
        virtual ~RSVertex(){};

        /** assignment operator */
        RSVertex& operator=(const RSVertex& rhs);
        /** test for equality */
        bool operator==(const RSVertex& rhs) const {
            return (idx == rhs.idx);
        };
        /** getter for the atom index */
        const unsigned int GetIndex() const {
            return idx;
        };
        /** getter for the edge pointer list */
        RSEdge* GetEdge(unsigned int eIdx) const {
            return edgeList.at(eIdx);
        };
        /** getter for the edge list size */
        const unsigned int GetEdgeCount() const {
            return (unsigned int)edgeList.size();
        };
        /** add edge */
        void AddEdge(RSEdge* edge) {
            // add the RS-edge-pointer to the edge list
            // DO NOT check if edge already exists
            // if( find( edgeList.begin(), edgeList.end(), eIdx) == edgeList.end() )
            edgeList.push_back(edge);
            // a atom which is part of the surface can't be buried
            buried = false;
        };
        /** remove edge from edge list */
        void RemoveEdge(RSEdge* edge);
        /** set atom index */
        void SetAtomIdx(unsigned int atomIdx) {
            idx = atomIdx;
            edgeList.clear();
        };
        /** set if the atom is burried */
        void SetAtomBuried(bool bury) {
            if (!edgeList.empty())
                return;
            else
                buried = bury;
        };
        /** returns true if the atom is burried, otherwise false */
        bool IsBuried() const {
            return buried;
        };
        void ClearEdgeList() {
            edgeList.clear();
            buried = false;
        };
        /** Get the atom position */
        const vislib::math::Vector<float, 3> GetPosition() const {
            return position;
        };
        /** Set the atom position */
        void SetPosition(vislib::math::Vector<float, 3> pos) {
            position = pos;
        };
        /** Get the atom radius */
        float GetRadius() {
            return radius;
        };
        /** Get the status of the RS-vertex */
        bool Treated() {
            return treated;
        };
        /** Set the RS-vertex as treated */
        void SetTreated() {
            treated = true;
        };
        /** Set the RS-vertex as not treated */
        void SetNotTreated() {
            treated = false;
        };
        /** Set texture coordinates for cutting planes */
        void SetTexCoord(unsigned int x, unsigned int y) {
            texCoordX = x;
            texCoordY = y;
        };
        /** Get texture coordinates for cutting planes */
        unsigned int GetTexCoordX() const {
            return texCoordX;
        };
        unsigned int GetTexCoordY() const {
            return texCoordY;
        };

    private:
        // the atom index of the corresponding atom in the protein data source
        unsigned int idx;
        // the indices of the edges to which this vertex belongs
        std::vector<RSEdge*> edgeList;
        // 'true' if the the atom is burried, viz. inside the surface
        bool buried;
        // atom position
        vislib::math::Vector<float, 3> position;
        // atom radius
        float radius;
        // 'true' if the RS-vertex was handled during RS-creation, else 'false'
        bool treated;
        // index for cutting planes texture access
        unsigned int texCoordX, texCoordY;
    };

    /**
     * Reduced Surface (RS) Edge class.
     * Stores pointers to its two vertex positions, pointers to the two faces
     * it belongs to and the center, radius and rotation angle of the torus.
     * A free RS-edge is indicated by both face pointers being null pointers (zero).
     */
    class RSEdge {
    public:
        RSEdge(RSVertex* v1, RSVertex* v2, vislib::math::Vector<float, 3> tCenter, float tRad);
        virtual ~RSEdge();

        RSEdge& operator=(const RSEdge& rhs);
        bool operator==(const RSEdge& rhs) const;
        bool SetRSFace(RSFace* f);
        void SetRSFaces(RSFace* f1, RSFace* f2) {
            face1 = f1;
            face2 = f2;
        };
        RSVertex* GetVertex1() const {
            return vert1;
        };
        RSVertex* GetVertex2() const {
            return vert2;
        };
        RSFace* GetFace1() const {
            return face1;
        };
        RSFace* GetFace2() const {
            return face2;
        };
        const vislib::math::Vector<float, 3>& GetTorusCenter() const {
            return torusCenter;
        };
        float GetTorusRadius() const {
            return torusRadius;
        };
        void SetRotationAngle(float angle) {
            rotationAngle = angle;
        };
        float GetRotationAngle() const {
            return rotationAngle;
        };
        // the list of probes (i.e. RS-faces) that cut this edge
        std::vector<RSFace*> cuttingProbes;
        void SetTexCoord(unsigned int x, unsigned int y) {
            texCoordX = x;
            texCoordY = y;
        };
        unsigned int GetTexCoordX() const {
            return texCoordX;
        };
        unsigned int GetTexCoordY() const {
            return texCoordY;
        };

    private:
        // the first vertex of the edge
        RSVertex* vert1;
        // the second vertex of the edge
        RSVertex* vert2;
        // the first face index
        RSFace* face1;
        // the second face index
        RSFace* face2;
        // torus center
        vislib::math::Vector<float, 3> torusCenter;
        // the radius of the torus
        float torusRadius;
        // the rotation angle for the torus cut
        float rotationAngle;
        // index for singularity texture access
        unsigned int texCoordX, texCoordY;
    };

    /**
     * Reduced Surface (RS) Face class.
     * Stores the indices of the vertex positions. The face is always a triangle.
     */
    class RSFace {
    public:
        /** ctor: takes three edges as input */
        RSFace(RSVertex* v1, RSVertex* v2, RSVertex* v3, RSEdge* e1, RSEdge* e2, RSEdge* e3,
            vislib::math::Vector<float, 3> norm, vislib::math::Vector<float, 3> pCenter);
        virtual ~RSFace();

        RSFace& operator=(const RSFace& rhs);
        bool operator==(const RSFace& rhs) const;
        const vislib::math::Vector<float, 3>& GetProbeCenter() const {
            return probeCenter;
        };
        const vislib::math::Vector<float, 3>& GetFaceNormal() const {
            return normal;
        };
        RSVertex* GetVertex1() const {
            return vert1;
        };
        RSVertex* GetVertex2() const {
            return vert2;
        };
        RSVertex* GetVertex3() const {
            return vert3;
        };
        RSEdge* GetEdge1() const {
            return edge1;
        };
        RSEdge* GetEdge2() const {
            return edge2;
        };
        RSEdge* GetEdge3() const {
            return edge3;
        };
        void SetEdge1(RSEdge* e) {
            edge1 = e;
        };
        void SetEdge2(RSEdge* e) {
            edge2 = e;
        };
        void SetEdge3(RSEdge* e) {
            edge3 = e;
        };
        void SetDualFace(RSFace* f) {
            dualFace = f;
        };
        RSFace* GetDualFace() {
            return dualFace;
        };
        void SetProbeIndex(unsigned int x, unsigned int y, unsigned int z) {
            probeIdx.Set(x, y, z);
        };
        const vislib::math::Vector<unsigned int, 3> GetProbeIndex() const {
            return probeIdx;
        };

        bool toDelete;

    private:
        // the first vertex of the face
        RSVertex* vert1;
        // the second vertex of the face
        RSVertex* vert2;
        // the third vertex of the face
        RSVertex* vert3;
        // the first edge of the face
        RSEdge* edge1;
        // the second edge of the face
        RSEdge* edge2;
        // the third edge of the face
        RSEdge* edge3;
        // face normal
        vislib::math::Vector<float, 3> normal;
        // probe center
        vislib::math::Vector<float, 3> probeCenter;
        // dual face with opposed normal (NULL if no such face exists
        RSFace* dualFace;
        // index of the probe in the probe voxel map
        vislib::math::Vector<unsigned int, 3> probeIdx;
    };

    /**
     * ctor
     * Computes the Reduced Surface(s) for the whole dataset provided by the
     * given MolecularDataCall.
     *
     * @param mol Pointer to the MolecularDataCall.
     * @param probeRad The radius of the probe.
     */
    ReducedSurface(megamol::protein_calls::MolecularDataCall* mol, float probeRad = 1.4f);

    /**
     * ctor
     * Computes the Reduced Surface for a specified amino acid chain of the
     * given MolecularDataCall.
     *
     * @param molId     Index of the molecule.
     * @param mol       Pointer to the MolecularDataCall.
     * @param probeRad  The radius of the probe.
     */
    ReducedSurface(unsigned int molId, megamol::protein_calls::MolecularDataCall* mol, float probeRad = 1.4f);

    /** dtor */
    virtual ~ReducedSurface();

    /** Get probe radius */
    const float GetProbeRadius() const {
        return probeRadius;
    };

    /**
     * Get the number of RS-vertices.
     * @return The number of RS-vertices.
     */
    unsigned int GetRSVertexCount() {
        return (unsigned int)rsVertex.size();
    };

    /**
     * Get the number of RS-edges.
     * @return The number of RS-edges.
     */
    unsigned int GetRSEdgeCount() {
        return (unsigned int)rsEdge.size();
    };

    /**
     * Get the number of RS-faces.
     * @return The number of RS-faces.
     */
    unsigned int GetRSFaceCount() {
        return (unsigned int)rsFace.size();
    };

    /**
     * Get a RS-vertex.
     * @param idx The index of the RS-vertex.
     * @return A pointer to the RS-vertex with index 'idx', if it exists, otherwise NULL
     */
    RSVertex* GetRSVertex(unsigned int idx) {
        if (rsVertex.size() > idx)
            return rsVertex[idx];
        else
            return NULL;
    };

    /**
     * Get a RS-edge.
     * @param idx The index of the RS-edge.
     * @return A pointer to the RS-edge with index 'idx', if it exists, otherwise NULL
     */
    RSEdge* GetRSEdge(unsigned int idx) {
        if (rsEdge.size() > idx)
            return rsEdge[idx];
        else
            return NULL;
    };

    /**
     * Get a RS-face.
     * @param idx The index of the RS-face.
     * @return A pointer to the RS-face with index 'idx', if it exists, otherwise NULL
     */
    RSFace* GetRSFace(unsigned int idx) {
        if (rsFace.size() > idx)
            return rsFace[idx];
        else
            return NULL;
    };

    /**
     * Get the number of RS-edges, which are cut by at least one probe.
     * @return The number of cut RS-edges.
     */
    unsigned int GetCutRSEdgesCount() {
        return countCutEdges;
    };

    /**
     * Read the next timestep and check for differences between the atoms.
     *
     * @param protein The protein data source.
     * @param lowerThreshold The lower treshold.
     * @param upperThreshold The upper treshold.
     */
    bool UpdateData(const float lowerThreshold, const float upperThreshold);

    /** compute the reduced surface of a molecule */
    void ComputeReducedSurface();

protected:
    /**
     * Write the indices of all atoms within the probe range relative to an atom at
     * position 'm' and radius 'rad' to the vicinity vector.
     * @param m The atom center.
     * @param rad The atom Radius.
     */
    void ComputeVicinity(vislib::math::Vector<float, 3> m, float rad);

    /**
     * Write the indices of all atoms that can be touched by the torus
     * definded a probe rotating around an edge to the vicinity vector.
     * @param edge The pointer to the edge.
     */
    void ComputeVicinityEdge(RSEdge* edge);

    /**
     * Write the indices of all atoms within the probe range relative to an
     * RS-vertex.
     * @param vertex The pointer to the RS-vertex.
     */
    void ComputeVicinityVertex(RSVertex* vertex);

    /**
     * Get the positions of all probes which cut a specific RS-edge.
     * @param edge The pointer to the edge.
     * @return A (possibly empty) vector of RS-faces which store the probe position.
     */
    std::vector<RSFace*> GetProbesCutEdge(RSEdge* edge);

    /**
     * Write the positions of all probes which cut a specific RS-edge.
     * @param edge The pointer to the edge.
     */
    void WriteProbesCutEdge(RSEdge* edge);

    bool SphereSphereIntersection(
        vislib::math::Vector<float, 3> m1, float rad1, vislib::math::Vector<float, 3> m2, float rad2);

    /**
     * Compute the two possible positions of the probe in contact with three atoms
     * and stores the first result in the RS-vertex, -edge and -face list.
     *
     * @param vI The pointer to the first atom.
     * @param vJ The pointer to the second atom.
     * @param vK The pointer to the third atom.
     * @return 'true' if possible positions were found, 'false' otherwise
     */
    bool ComputeFirstFixedProbePos(RSVertex* vI, RSVertex* vJ, RSVertex* vK);

    /**
     * Find the first RS-face starting from RS-vertex 'vertex'.
     * The results are stored in the appropriate containers for RS-vertices,
     * -edges and -faces.
     *
     * @param vertex The pointer to the first RS-vertex.
     * @return 'true' if the first RS-face was found, 'false' otherwise.
     */
    bool FindFirstRSFace(RSVertex* vertex);

    /**
     * Compute the next RS-face for the given edge.
     * The edge must have one face assigned.
     *
     * @param edgeIdx The index of the edge.
     */
    void ComputeRSFace(unsigned int edgeIdx);

    /**
     * Compute the rotation angle between two probe positions for a given direction
     * of rotation.
     *
     * @param tCenter The center of the torus.
     * @param n1 The normal of the plane pointing in the direction of the rotation.
     * @param pPos The new probe position.
     * @param pPosOld The old probe position.
     *
     * @return The angle between the two probes.
     */
    float ComputeAngleBetweenProbes(vislib::math::Vector<float, 3> tCenter, vislib::math::Vector<float, 3> n1,
        vislib::math::Vector<float, 3> pPos, vislib::math::Vector<float, 3> pPosOld);

    /**
     * Compute the singularities for every RS-edge (store the positions of the
     * probes that cut it).
     */
    void ComputeSingularities();

    /**
     * Search all RS-faces whose probe is cut by the given RS-vertex.
     * @param vertex The pointer to the RS-vertex.
     */
    void ComputeProbeCutVertex(RSVertex* vertex);

private:
    // The pointer to the protein data interface
    megamol::protein_calls::MolecularDataCall* molecule;

    // Boolean flag for global (true) or single chain (false) computation
    const bool globalRS;

    // the first atom index
    unsigned int firstAtomIdx;
    // the total number of atoms
    unsigned int numberOfAtoms;

    // zero vector in R^3
    const vislib::math::Vector<float, 3> zeroVec3;

    // the bounding box of the protein
    vislib::math::Cuboid<float> bBox;

    // epsilon value for float-comparison
    float epsilon;

    // radius of the probe atom
    float probeRadius;

    // auxiliary arrays
    std::vector<RSVertex*> vicinity;
    std::vector<RSFace*> cutFaces;

    // the RS-vertex list
    std::vector<RSVertex*> rsVertex;
    // the RS-edge list
    std::vector<RSEdge*> rsEdge;
    // the RS-face list
    std::vector<RSFace*> rsFace;

    // vector for the voxel map for RS-vertex positions
    std::vector<std::vector<std::vector<std::vector<RSVertex*>>>> voxelMap;
    // vector for the voxel map for probe positions
    std::vector<std::vector<std::vector<std::vector<RSFace*>>>> voxelMapProbes;
    // float voxel length
    float voxelLength;

    std::vector<float> atoms;

    // number of RS-edges, which are cut by at least one probe
    unsigned int countCutEdges;
};

} // namespace megamol::protein

#endif /* MEGAMOL_REDUCEDSURFACE_H_INCLUDED */
