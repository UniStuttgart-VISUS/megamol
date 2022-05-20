/*
 * MapGenerator.cpp
 * Copyright (C) 2006-2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "Octree.h"

using namespace megamol;
using namespace megamol::molecularmaps;


/*
 * Octree::~Octree
 */
Octree::~Octree() {
    this->face_bboxs.erase(this->face_bboxs.begin(), this->face_bboxs.end());
    this->face_bboxs.clear();
}


/*
 * Octree::createFaceBoundingBoxes
 */
size_t Octree::ConvertToCUDAOctree(
    std::vector<CudaOctreeNode>& p_octree_nodes, std::vector<std::vector<uint>>& p_node_faces) {
    // Convert the Octree nodes.
    p_octree_nodes = std::vector<CudaOctreeNode>(this->cuda_node_cnt + 1);
    p_node_faces = std::vector<std::vector<uint>>(this->cuda_node_cnt + 1);
    std::vector<const OctreeNode*> queue = std::vector<const OctreeNode*>(this->node_cnt);
    size_t queue_idx = 0;
    queue[queue_idx++] = &this->root;

    // Get every node on the tree and copy it to the CUDA representation.
    while (queue_idx != 0) {
        // Get the current node.
        auto curr = queue[--queue_idx];

        // Create the node in CUDA representation.
        p_octree_nodes[curr->cuda_idx] = CudaOctreeNode(CudaBoundingBox(curr->bounding_box.max, curr->bounding_box.min),
            static_cast<uint>(curr->children.size()), static_cast<uint>(curr->faces.size()));

        // Add the faces to the node.
        p_node_faces[curr->cuda_idx] = curr->faces;

        // Add the children of the current node to the queue.
        for (const auto& node : curr->children) {
            queue[queue_idx++] = &node;
        }
    }

    // Return the number of nodes in the Octree.
    return this->node_cnt;
}


/*
 * Octree::createFaceBoundingBoxes
 */
void Octree::createFaceBoundingBoxes(const std::vector<uint>& p_faces, const std::vector<float>& p_vertices) {
    uint3 face_idxs;
    size_t size;
    std::vector<float3> vertices;

    // Initialise the bounding boxes.
    size = p_faces.size() / 3;
    this->face_bboxs.erase(this->face_bboxs.begin(), this->face_bboxs.end());
    this->face_bboxs.clear();
    this->face_bboxs.reserve(size);

    // Get the vertices of each face an determine the bounding box of these vertices.
    vertices = std::vector<float3>(3);
    for (size_t i = 0; i < size; i++) {
        face_idxs = make_uint3(p_faces[i * 3], p_faces[i * 3 + 1], p_faces[i * 3 + 2]);
        vertices[0] =
            make_float3(p_vertices[face_idxs.x * 3], p_vertices[face_idxs.x * 3 + 1], p_vertices[face_idxs.x * 3 + 2]);
        vertices[1] =
            make_float3(p_vertices[face_idxs.y * 3], p_vertices[face_idxs.y * 3 + 1], p_vertices[face_idxs.y * 3 + 2]);
        vertices[2] =
            make_float3(p_vertices[face_idxs.z * 3], p_vertices[face_idxs.z * 3 + 1], p_vertices[face_idxs.z * 3 + 2]);
        this->face_bboxs.push_back(BoundingBox(vertices));
    }
    this->face_bboxs.shrink_to_fit();
}


/*
 * Octree::createOctree
 */
void Octree::createOctree(const float3& p_min_dim, OctreeNode& p_node) {
    float3 center, dim, half, max, min, zero;
    std::vector<BoundingBox> octant;
    std::vector<std::vector<uint>> node_faces;
    std::vector<bool> to_delete;
    size_t idx;

    // Increase the cuda node count.
    if (p_node.cuda_idx > this->cuda_node_cnt) {
        this->cuda_node_cnt = p_node.cuda_idx;
    }

    // If the node does not contain more than one face it is a leaf.
    if (p_node.faces.size() <= 1) {
        return;
    }

    // Determine the bounding box of the current node.
    zero.x = 0.0f;
    zero.y = 0.0f;
    zero.z = 0.0f;

    max = p_node.bounding_box.max;
    min = p_node.bounding_box.min;
    dim = max - min;
    // The bounding box of the node is too small, therefor the node is a leaf.
    if (dim.x == zero.x && dim.y == zero.y && dim.z == zero.z) {
        return;
    }
    if (dim.x <= p_min_dim.x && dim.y <= p_min_dim.y && dim.z <= p_min_dim.z) {
        return;
    }

    // Create new bounding boxes for each child node.
    half = dim / 2.0f;
    center = min + half;
    octant = std::vector<BoundingBox>(8);
    octant[0] = BoundingBox(center, min);
    octant[1] = BoundingBox(make_float3(max.x, center.y, center.z), make_float3(center.x, min.y, min.z));
    octant[2] = BoundingBox(make_float3(max.x, center.y, max.z), make_float3(center.x, min.y, center.z));
    octant[3] = BoundingBox(make_float3(center.x, center.y, max.z), make_float3(min.x, min.y, center.z));
    octant[4] = BoundingBox(make_float3(center.x, max.y, center.z), make_float3(min.x, center.y, min.z));
    octant[5] = BoundingBox(make_float3(max.x, max.y, center.z), make_float3(center.x, center.y, min.z));
    octant[6] = BoundingBox(max, center);
    octant[7] = BoundingBox(make_float3(center.x, max.y, max.z), make_float3(min.x, center.y, center.z));

    // Add the faces to the correct child node.
    node_faces = std::vector<std::vector<uint>>(8);
    for (uint i = 0; i < 8; i++) {
        node_faces[i].reserve(p_node.faces.size());
    }
    to_delete = std::vector<bool>(p_node.faces.size() + 1, false);

    idx = 0;
    for (const auto face : p_node.faces) {
        for (size_t i = 0; i < octant.size(); i++) {
            if (this->face_bboxs[face].IsContained(octant[i])) {
                node_faces[i].push_back(face);
                to_delete[idx] = true;
                break;
            }
        }
        idx++;
    }

    // Remove the faces from the current node.
    p_node.RemoveFaces(to_delete);

    // Add the new node as a child to the current node.
    OctreeNode new_node;
    for (uint i = 0; i < 8; i++) {
        new_node = OctreeNode(octant[i], (p_node.cuda_idx * 8) + (i + 1), node_faces[i]);
        p_node.children.push_back(new_node);
        this->node_cnt++;
        createOctree(p_min_dim, p_node.children[i]);
    }
}


/*
 * Octree::createOctreeRootNode
 */
void Octree::CreateOctreeRootNode(const std::vector<uint>& p_faces, const float3& p_max, const float3& p_min,
    const float3& p_min_dim, const std::vector<float>& p_vertices) {
    // Create the root node, it contains all faces.
    this->root = OctreeNode(BoundingBox(p_max, p_min), 0, p_faces.size() / 3);

    // Compute the bounding boxes for all faces of the surface.
    createFaceBoundingBoxes(p_faces, p_vertices);

    // Compute the full Octree.
    this->node_cnt = 1;
    this->cuda_node_cnt = 1;
    createOctree(p_min_dim, this->root);
}


/*
 * Octree::IntersectOctree
 */
int Octree::IntersectOctree(const std::vector<uint>& p_faces, const Ray& p_ray, const std::vector<float>& p_vertices) {
    // Initialise the queue for the intersection tests.
    vec3ui face_idxs;
    std::vector<const OctreeNode*> queue = std::vector<const OctreeNode*>(this->node_cnt);
    std::vector<vec3f> vertices;
    size_t queue_idx = 0;

    // Add the root to the queue.
    queue[queue_idx++] = &this->root;
    vertices = std::vector<vec3f>(3);

    // Get the current Octree node from the queue and test for intersections.
    while (queue_idx != 0) {
        auto curr = queue[--queue_idx];

        // Check all faces that belong to the current node.
        for (const auto face : curr->faces) {
            face_idxs = vec3ui(p_faces[face * 3], p_faces[face * 3 + 1], p_faces[face * 3 + 2]);
            vertices[0] = vec3f(p_vertices[face_idxs.GetX() * 3], p_vertices[face_idxs.GetX() * 3 + 1],
                p_vertices[face_idxs.GetX() * 3 + 2]);
            vertices[1] = vec3f(p_vertices[face_idxs.GetY() * 3], p_vertices[face_idxs.GetY() * 3 + 1],
                p_vertices[face_idxs.GetY() * 3 + 2]);
            vertices[2] = vec3f(p_vertices[face_idxs.GetZ() * 3], p_vertices[face_idxs.GetZ() * 3 + 1],
                p_vertices[face_idxs.GetZ() * 3 + 2]);
            if (this->rayTriangleIntersection(p_ray, vertices)) {
                // Found intersection return one of the three vertex IDs.
                return static_cast<int>(face_idxs.GetX());
            }
        }

        // Add all children of the current node to the queue if the ray intersects their bounding box.
        for (const auto& node : curr->children) {
            if (node.bounding_box.RayIntersection(p_ray)) {
                queue[queue_idx++] = &node;
            }
        }
    }

    // No intersection found, return -1.
    return -1;
}


/*
 * Octree::IntersectOctree
 */
int Octree::IntersectOctree(
    const std::vector<uint>& p_faces, const Ray& p_ray, const std::vector<float>& p_vertices, uint& p_face_id) {
    // Initialise the queue for the intersection tests.
    vec3ui face_idxs;
    std::vector<const OctreeNode*> queue = std::vector<const OctreeNode*>(this->node_cnt);
    std::vector<vec3f> vertices;
    size_t queue_idx = 0;
    int retval = -1;
    float min_dist = std::numeric_limits<float>::max();

    // Add the root to the queue.
    queue[queue_idx++] = &this->root;
    vertices = std::vector<vec3f>(3);

    // Get the current Octree node from the queue and test for intersections.
    while (queue_idx != 0) {
        auto curr = queue[--queue_idx];

        // Check all faces that belong to the current node.
        for (const auto face : curr->faces) {
            face_idxs = vec3ui(p_faces[face * 3], p_faces[face * 3 + 1], p_faces[face * 3 + 2]);
            vertices[0] = vec3f(p_vertices[face_idxs.GetX() * 3], p_vertices[face_idxs.GetX() * 3 + 1],
                p_vertices[face_idxs.GetX() * 3 + 2]);
            vertices[1] = vec3f(p_vertices[face_idxs.GetY() * 3], p_vertices[face_idxs.GetY() * 3 + 1],
                p_vertices[face_idxs.GetY() * 3 + 2]);
            vertices[2] = vec3f(p_vertices[face_idxs.GetZ() * 3], p_vertices[face_idxs.GetZ() * 3 + 1],
                p_vertices[face_idxs.GetZ() * 3 + 2]);
            if (this->rayTriangleIntersection(p_ray, vertices)) {
                // Found intersection check if the distance is smaller than the closest intersection
                // so far.
                float x = vertices[0].GetX() - p_ray.origin.GetX();
                float y = vertices[0].GetY() - p_ray.origin.GetY();
                float z = vertices[0].GetZ() - p_ray.origin.GetZ();
                float dist = x * x + y * y + z * z;
                if (dist < min_dist) {
                    min_dist = dist;
                    p_face_id = face;
                    retval = face_idxs.GetX();
                }
            }
        }

        // Add all children of the current node to the queue if the ray intersects their bounding box.
        for (const auto& node : curr->children) {
            if (node.bounding_box.RayIntersection(p_ray)) {
                queue[queue_idx++] = &node;
            }
        }
    }

    // Return the intersection or -1 if no intersection was found.
    return retval;
}


/*
 * Octree::Octree
 */
Octree::Octree() {
    this->face_bboxs = std::vector<BoundingBox>(0);
    this->root = OctreeNode();
}

/*
 * Octree::RadiusSearch
 */
bool Octree::RadiusSearch(const std::vector<uint>& p_faces, const vec4d& p_querySphere,
    const std::vector<float>& p_vertices, std::vector<uint>& p_resultFaceIndices) {
    // Initialise the queue for the intersection tests.
    vec3ui face_idxs;
    std::vector<const OctreeNode*> queue = std::vector<const OctreeNode*>(this->node_cnt);
    std::vector<vec3f> vertices;
    size_t queue_idx = 0;

    // TODO check memory allocation to speed this up
    p_resultFaceIndices.clear();

    queue[queue_idx++] = &this->root;
    vertices = std::vector<vec3f>(3);

    while (queue_idx != 0) {
        auto curr = queue[--queue_idx];

        // Check all faces that belong to the current node.
        for (const auto face : curr->faces) {
            face_idxs = vec3ui(p_faces[face * 3], p_faces[face * 3 + 1], p_faces[face * 3 + 2]);
            vertices[0] = vec3f(p_vertices[face_idxs.GetX() * 3], p_vertices[face_idxs.GetX() * 3 + 1],
                p_vertices[face_idxs.GetX() * 3 + 2]);
            vertices[1] = vec3f(p_vertices[face_idxs.GetY() * 3], p_vertices[face_idxs.GetY() * 3 + 1],
                p_vertices[face_idxs.GetY() * 3 + 2]);
            vertices[2] = vec3f(p_vertices[face_idxs.GetZ() * 3], p_vertices[face_idxs.GetZ() * 3 + 1],
                p_vertices[face_idxs.GetZ() * 3 + 2]);
            if (this->sphereTriangleIntersection(p_querySphere, vertices)) {
                // Found intersection, add it to the result.
                p_resultFaceIndices.push_back(face);
            }
        }
        // Add all children of the current node to the queue if the sphere intersects their bounding box.
        for (const auto& node : curr->children) {
            if (node.bounding_box.SphereIntersection(p_querySphere)) {
                queue[queue_idx++] = &node;
            }
        }
    }

    return (p_resultFaceIndices.size() > 0);
}


/*
 * Octree::rayTriangleIntersection
 */
bool Octree::rayTriangleIntersection(const Ray& p_ray, const std::vector<vec3f>& p_vertices) {
    vec3f e2 = p_vertices[2] - p_vertices[0];
    vec3f e1 = p_vertices[1] - p_vertices[0];
    vec3f r = p_ray.dir.Cross(e2);
    vec3f s = p_ray.origin - p_vertices[0];
    float denom = e1.Dot(r);
    if (abs(denom) < 1e-5)
        return false;
    float f = 1.0f / denom;
    vec3f q = s.Cross(e1);
    float u = s.Dot(r);

    if (denom > 1e-5) {
        if (u < 0.0f || u > denom)
            return false;
        float v = p_ray.dir.Dot(q);
        if (v < 0.0f || (u + v) > denom)
            return false;

    } else {
        if (u > 0.0f || u < denom)
            return false;
        float v = p_ray.dir.Dot(q);
        if (v > 0.0f || (u + v) < denom)
            return false;
    }

    float t = f * e2.Dot(q);
    if (t > 1e-5)
        return true;
    else
        return false;
}


/*
 * Octree::sphereTriangleIntersection
 */
bool Octree::sphereTriangleIntersection(const vec4d& p_sphere, const std::vector<vec3f>& p_vertices) {
    double radiusSquared = p_sphere[3] * p_sphere[3];
    double squaredDist = 0.0;
    for (const auto& vert : p_vertices) {
        squaredDist = (p_sphere.GetX() - vert.GetX()) * (p_sphere.GetX() - vert.GetX()) +
                      (p_sphere.GetY() - vert.GetY()) * (p_sphere.GetY() - vert.GetY()) +
                      (p_sphere.GetZ() - vert.GetZ()) * (p_sphere.GetZ() - vert.GetZ());
        if (squaredDist <= radiusSquared)
            return true;
    }
    return false;
}
