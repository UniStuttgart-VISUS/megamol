/*
 * MapGenerator.h
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMMOLMAPPLG_OCTREE_H_INCLUDED
#define MMMOLMAPPLG_OCTREE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Types.h"
#include <vector>

namespace megamol {
namespace molecularmaps {

class Octree {
public:
    /**
     * Destroys the instance.
     */
    ~Octree();

    /**
     * Converts the Octree tot he CUDA format.
     *
     * @param p_octree_nodes will contain the Octree nodes in the CUDA format
     * @param p_face_bbox will contain the bounding boxes of the faces in the
     * CUDA format
     *
     * @return the Octree in a format that CUDA understands.
     */
    size_t ConvertToCUDAOctree(
        std::vector<CudaOctreeNode>& p_octree_nodes, std::vector<std::vector<uint>>& p_node_faces);

    /**
     * Create the root node of the Octree for the surface based on the faces.
     * Also creates the bounding boxes for all faces of the surface. Then
     * creates the full octree.
     *
     * @param p_faces the faces of the surface
     * @param p_max the right, back, top point of the bounding box that
     * contains every face
     * @param p_min the left, front, bottom point of the bounding box that
     * contains every face
     * @param the minimum dimension of the bounding box, if the bounding box
     * of a node is smaller it is a leaf node
     * @param p_vertices the vertices of the surface
     */
    void CreateOctreeRootNode(const std::vector<uint>& p_faces, const float3& p_max, const float3& p_min,
        const float3& p_min_dim, const std::vector<float>& p_vertices);

    /**
     * Find the face that is hitten by the given ray. Traverse the Octree to
     * find the first intersection.
     *
     * @param p_faces the faces of the surface
     * @param p_ray the ray with an origin and direction
     * @param p_vertices the vertices of the surface
     *
     * @return the ID of the face that was hit or -1 if no intersection was
     * found
     */
    int IntersectOctree(const std::vector<uint>& p_faces, const Ray& p_ray, const std::vector<float>& p_vertices);

    /**
     * Find the face that is hitten by the given ray. Traverse the Octree to
     * find the closest intersection.
     *
     * @param p_faces the faces of the surface
     * @param p_ray the ray with an origin and direction
     * @param p_vertices the vertices of the surface
     * @param p_face_id the face of the surface that is intersected
     *
     * @return the ID of the face that was hit or -1 if no intersection was
     * found
     */
    int IntersectOctree(
        const std::vector<uint>& p_faces, const Ray& p_ray, const std::vector<float>& p_vertices, uint& p_face_id);

    /**
     * Initialises an empty instance.
     */
    Octree(void);

    /**
     * Find all faces that that lie in or intersect a sphere with a given radius and
     * position.
     *
     * @param p_faces The faces of the surface
     * @param p_querySphere The sphere which radius and position are used for the query
     * @param p_vertices The vertices of the surface
     * @param p_resultFaceIndices Vector containing the indices of the found faces
     *
     * @return True, if at least one face lies partially or totally inside the query sphere.
     *         False otherwise.
     */
    bool RadiusSearch(const std::vector<uint>& p_faces, const vec4d& p_querySphere,
        const std::vector<float>& p_vertices, std::vector<uint>& p_resultFaceIndices);

private:
    /**
     * Create the bounding boxes for each face.
     *
     * @param p_faces the faces of the surface
     * @param p_vertices the vertices of the surface
     */
    void createFaceBoundingBoxes(const std::vector<uint>& p_faces, const std::vector<float>& p_vertices);

    /**
     * Create the Octree based on the faces of the new surface.
     *
     * @param p_min_dim the minimum dimension of the bounding box, if the
     * bounding box of a node is smaller it is a leaf node
     * @param p_cuda_node the current node in the Octree in the CUDA format
     * @param p_node the current node in the Octree
     */
    void createOctree(const float3& p_min_dim, OctreeNode& p_node);

    /**
     * Check if the ray intersects the triangle based on the vertices.
     *
     * @param p_ray the ray with origin an direction
     * @param p_vertices the vertices of the triangle
     *
     * @return true if the ray intersects, false otherwise
     */
    bool rayTriangleIntersection(const Ray& p_ray, const std::vector<vec3f>& p_vertices);

    /**
     * Check if the sphere contains at least a part of the triangle.
     *
     * @param p_sphere the sphere with position and radius.
     * @param p_vertices the vertices of the triangle.
     *
     * @return true if the sphere intersects the triangle, false otherwise.
     */
    bool sphereTriangleIntersection(const vec4d& p_sphere, const std::vector<vec3f>& p_vertices);

    /** The amount of CUDA nodes we need to represent the Octree. */
    size_t cuda_node_cnt;

    /** The bounding boxes for all faces. */
    std::vector<BoundingBox> face_bboxs;

    /** The number of nodes in the Octree. */
    size_t node_cnt;

    /** The root node of the Octree. */
    OctreeNode root;
};

} /* end namespace molecularmaps */
} /* end namespace megamol */

#endif
