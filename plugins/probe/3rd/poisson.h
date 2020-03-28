/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#pragma once

#include "common.h"
#include "poisson4/function_data.h"
#include "poisson4/geometry.h"
#include "poisson4/multi_grid_octree_data.h"
#include "poisson4/octree_poisson.h"
#include "poisson4/ppolynomial.h"
#include "poisson4/sparse_matrix.h"
#include "reconstruction.h"

namespace pcl {
namespace poisson {
class CoredVectorMeshData;
template <class Real> struct Point3D;
} // namespace poisson

/** \brief The Poisson surface reconstruction algorithm.
 * \note Code adapted from Misha Kazhdan: http://www.cs.jhu.edu/~misha/Code/PoissonRecon/
 * \note Based on the paper:
 *       * Michael Kazhdan, Matthew Bolitho, Hugues Hoppe, "Poisson surface reconstruction",
 *         SGP '06 Proceedings of the fourth Eurographics symposium on Geometry processing
 * \author Alexandru-Eugen Ichim
 * \ingroup surface
 */
template <typename PointNT> class Poisson : public SurfaceReconstruction<PointNT> {
public:
    using Ptr = std::shared_ptr<Poisson<PointNT>>;
    using ConstPtr = std::shared_ptr<const Poisson<PointNT>>;

    using SurfaceReconstruction<PointNT>::input_;
    using SurfaceReconstruction<PointNT>::tree_;

    using PointCloud = PointCloud<PointNT>;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;


    using KdTree = pcl::KdTree<PointNT>;
    using KdTreePtr = typename KdTree::Ptr;

    /** \brief Constructor that sets all the parameters to working default values. */
    Poisson();

    /** \brief Destructor. */
    ~Poisson();

    /** \brief Create the surface.
     * \param[out] points the vertex positions of the resulting mesh
     * \param[out] polygons the connectivity of the resulting mesh
     */
    void performReconstruction(pcl::PointCloud<PointNT>& points, std::vector<pcl::Vertices>& polygons) override;

    /** \brief Set the maximum depth of the tree that will be used for surface reconstruction.
     * \note Running at depth d corresponds to solving on a voxel grid whose resolution is no larger than
     * 2^d x 2^d x 2^d. Note that since the reconstructor adapts the octree to the sampling density, the specified
     * reconstruction depth is only an upper bound.
     * \param[in] depth the depth parameter
     */
    inline void setDepth(int depth) { depth_ = depth; }

    /** \brief Get the depth parameter */
    inline int getDepth() { return depth_; }

    inline void setMinDepth(int min_depth) { min_depth_ = min_depth; }

    inline int getMinDepth() { return min_depth_; }

    inline void setPointWeight(float point_weight) { point_weight_ = point_weight; }

    inline float getPointWeight() { return point_weight_; }

    /** \brief Set the ratio between the diameter of the cube used for reconstruction and the diameter of the
     * samples' bounding cube.
     * \param[in] scale the given parameter value
     */
    inline void setScale(float scale) { scale_ = scale; }

    /** Get the ratio between the diameter of the cube used for reconstruction and the diameter of the
     * samples' bounding cube.
     */
    inline float getScale() { return scale_; }

    /** \brief Set the the depth at which a block Gauss-Seidel solver is used to solve the Laplacian equation
     * \note Using this parameter helps reduce the memory overhead at the cost of a small increase in
     * reconstruction time. (In practice, we have found that for reconstructions of depth 9 or higher a subdivide
     * depth of 7 or 8 can greatly reduce the memory usage.)
     * \param[in] solver_divide the given parameter value
     */
    inline void setSolverDivide(int solver_divide) { solver_divide_ = solver_divide; }

    /** \brief Get the depth at which a block Gauss-Seidel solver is used to solve the Laplacian equation */
    inline int getSolverDivide() { return solver_divide_; }

    /** \brief Set the depth at which a block iso-surface extractor should be used to extract the iso-surface
     * \note Using this parameter helps reduce the memory overhead at the cost of a small increase in extraction
     * time. (In practice, we have found that for reconstructions of depth 9 or higher a subdivide depth of 7 or 8
     * can greatly reduce the memory usage.)
     * \param[in] iso_divide the given parameter value
     */
    inline void setIsoDivide(int iso_divide) { iso_divide_ = iso_divide; }

    /** \brief Get the depth at which a block iso-surface extractor should be used to extract the iso-surface */
    inline int getIsoDivide() { return iso_divide_; }

    /** \brief Set the minimum number of sample points that should fall within an octree node as the octree
     * construction is adapted to sampling density
     * \note For noise-free samples, small values in the range [1.0 - 5.0] can be used. For more noisy samples,
     * larger values in the range [15.0 - 20.0] may be needed to provide a smoother, noise-reduced, reconstruction.
     * \param[in] samples_per_node the given parameter value
     */
    inline void setSamplesPerNode(float samples_per_node) { samples_per_node_ = samples_per_node; }

    /** \brief Get the minimum number of sample points that should fall within an octree node as the octree
     * construction is adapted to sampling density
     */
    inline float getSamplesPerNode() { return samples_per_node_; }

    /** \brief Set the confidence flag
     * \note Enabling this flag tells the reconstructor to use the size of the normals as confidence information.
     * When the flag is not enabled, all normals are normalized to have unit-length prior to reconstruction.
     * \param[in] confidence the given flag
     */
    inline void setConfidence(bool confidence) { confidence_ = confidence; }

    /** \brief Get the confidence flag */
    inline bool getConfidence() { return confidence_; }

    /** \brief Enabling this flag tells the reconstructor to output a polygon mesh (rather than triangulating the
     * results of Marching Cubes).
     * \param[in] output_polygons the given flag
     */
    inline void setOutputPolygons(bool output_polygons) { output_polygons_ = output_polygons; }

    /** \brief Get whether the algorithm outputs a polygon mesh or a triangle mesh */
    inline bool getOutputPolygons() { return output_polygons_; }

    /** \brief Set the degree parameter
     * \param[in] degree the given degree
     */
    inline void setDegree(int degree) { degree_ = degree; }

    /** \brief Get the degree parameter */
    inline int getDegree() { return degree_; }

    /** \brief Set the manifold flag.
     * \note Enabling this flag tells the reconstructor to add the polygon barycenter when triangulating polygons
     * with more than three vertices.
     * \param[in] manifold the given flag
     */
    inline void setManifold(bool manifold) { manifold_ = manifold; }

    /** \brief Get the manifold flag */
    inline bool getManifold() { return manifold_; }

    /** \brief Sets data input
     *
     * \param the input cloud
     */
    inline void setInputCloud(const PointCloudConstPtr& input) override { this->input_ = input; }

    //inline void setIndices(const IndicesConstPtr& indices) override { this->indices_ = indices; }

    inline void setIndices(const IndicesPtr& indices) override { this->indices_ = indices; }

protected:
    /** \brief Class get name method. */
    std::string getClassName() const override { return ("Poisson"); }

private:
    int depth_;
    int min_depth_;
    float point_weight_;
    float scale_;
    int solver_divide_;
    int iso_divide_;
    float samples_per_node_;
    bool confidence_;
    bool output_polygons_;

    bool no_reset_samples_;
    bool no_clip_tree_;
    bool manifold_;

    int refine_;
    int kernel_depth_;
    int degree_;
    bool non_adaptive_weights_;
    bool show_residual_;
    int min_iterations_;
    float solver_accuracy_;

    template <int Degree>
    void execute(poisson::CoredVectorMeshData& mesh, poisson::Point3D<float>& translate, float& scale);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointNT>
pcl::Poisson<PointNT>::Poisson()
    : depth_(8)
    , min_depth_(5)
    , point_weight_(4)
    , scale_(1.1f)
    , solver_divide_(8)
    , iso_divide_(8)
    , samples_per_node_(1.0)
    , confidence_(false)
    , output_polygons_(false)
    , no_reset_samples_(false)
    , no_clip_tree_(false)
    , manifold_(true)
    , refine_(3)
    , kernel_depth_(8)
    , degree_(2)
    , non_adaptive_weights_(false)
    , show_residual_(false)
    , min_iterations_(8)
    , solver_accuracy_(1e-3f) {}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointNT> pcl::Poisson<PointNT>::~Poisson() {}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointNT>
template <int Degree>
void pcl::Poisson<PointNT>::execute(poisson::CoredVectorMeshData& mesh, poisson::Point3D<float>& center, float& scale) {
    pcl::poisson::Real iso_value = 0;
    poisson::TreeNodeData::UseIndex = 1;
    poisson::Octree<Degree> tree;

    /// TODO OPENMP stuff
    //    tree.threads = Threads.value;
    center.coords[0] = center.coords[1] = center.coords[2] = 0;


    if (solver_divide_ < min_depth_) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[pcl::Poisson] solver_divide_ must be at least as large as min_depth_: %d >= %d\n", solver_divide_,
            min_depth_);
        solver_divide_ = min_depth_;
    }
    if (iso_divide_ < min_depth_) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[pcl::Poisson] iso_divide_ must be at least as large as min_depth_: %d >= %d\n", iso_divide_, min_depth_);
        iso_divide_ = min_depth_;
    }

    pcl::poisson::TreeOctNode::SetAllocator(MEMORY_ALLOCATOR_BLOCK_SIZE);

    kernel_depth_ = depth_ - 2;

    tree.setBSplineData(depth_, pcl::poisson::Real(1.0 / (1 << depth_)), true);

    tree.maxMemoryUsage = 0;


    int point_count = tree.template setTree<PointNT>(input_, depth_, min_depth_, kernel_depth_, samples_per_node_,
        scale_, center, scale, confidence_, point_weight_, !non_adaptive_weights_);

    tree.ClipTree();
    tree.finalize();
    tree.RefineBoundary(iso_divide_);

    PCL_DEBUG("Input Points: %d\n", point_count);
    PCL_DEBUG("Leaves/Nodes: %d/%d\n", tree.tree.leaves(), tree.tree.nodes());

    tree.maxMemoryUsage = 0;
    tree.SetLaplacianConstraints();

    tree.maxMemoryUsage = 0;
    tree.LaplacianMatrixIteration(solver_divide_, show_residual_, min_iterations_, solver_accuracy_);

    iso_value = tree.GetIsoValue();

    tree.GetMCIsoTriangles(iso_value, iso_divide_, &mesh, 0, 1, manifold_, output_polygons_);
}


template <typename PointNT>
void pcl::Poisson<PointNT>::performReconstruction(
    pcl::PointCloud<PointNT>& points, std::vector<pcl::Vertices>& polygons) {
    poisson::CoredVectorMeshData mesh;
    poisson::Point3D<float> center;
    float scale = 1.0f;

    switch (degree_) {
    case 1: {
        execute<1>(mesh, center, scale);
        break;
    }
    case 2: {
        execute<2>(mesh, center, scale);
        break;
    }
    case 3: {
        execute<3>(mesh, center, scale);
        break;
    }
    case 4: {
        execute<4>(mesh, center, scale);
        break;
    }
    case 5: {
        execute<5>(mesh, center, scale);
        break;
    }
    default: { vislib::sys::Log::DefaultLog.WriteError("Degree %d not supported\n", degree_); }
    }

    // Write output PolygonMesh
    // Write vertices
    points.points.resize(int(mesh.outOfCorePointCount() + mesh.inCorePoints.size()));
    poisson::Point3D<float> p;
    for (int i = 0; i < int(mesh.inCorePoints.size()); i++) {
        p = mesh.inCorePoints[i];
        points.points[i].x = p.coords[0] * scale + center.coords[0];
        points.points[i].y = p.coords[1] * scale + center.coords[1];
        points.points[i].z = p.coords[2] * scale + center.coords[2];
    }
    for (int i = int(mesh.inCorePoints.size()); i < int(mesh.outOfCorePointCount() + mesh.inCorePoints.size()); i++) {
        mesh.nextOutOfCorePoint(p);
        points.points[i].x = p.coords[0] * scale + center.coords[0];
        points.points[i].y = p.coords[1] * scale + center.coords[1];
        points.points[i].z = p.coords[2] * scale + center.coords[2];
    }

    polygons.resize(mesh.polygonCount());

    // Write faces
    std::vector<poisson::CoredVertexIndex> polygon;
    for (int p_i = 0; p_i < mesh.polygonCount(); p_i++) {
        pcl::Vertices v;
        mesh.nextPolygon(polygon);
        assert(polygon.size() == 3);
        //v.vertices.resize(polygon.size());

        for (int i = 0; i < static_cast<int>(polygon.size()); ++i)
            if (polygon[i].inCore)
                v.vertices[i] = polygon[i].idx;
            else
                v.vertices[i] = polygon[i].idx + int(mesh.inCorePoints.size());

        polygons[p_i] = v;
    }
}


#define PCL_INSTANTIATE_Poisson(T) template class PCL_EXPORTS pcl::Poisson<T>;

} // namespace pcl
