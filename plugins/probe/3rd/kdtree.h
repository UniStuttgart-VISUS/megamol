/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
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
 *   * Neither the name of the copyright holder(s) nor the names of its
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
 *
 */

#pragma once

#include "common.h"
#include "nanoflann.hpp"


namespace pcl
{

/** \brief KdTree represents the base spatial locator class for kd-tree implementations.
 * \author Radu B Rusu, Bastian Steder, Michael Dixon
 * \ingroup kdtree
 */
template <typename PointT> class KdTree {
public:
    using IndicesPtr = std::shared_ptr<std::vector<int>>;
    using IndicesConstPtr = std::shared_ptr<const std::vector<uint32_t>>;

    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = std::shared_ptr<PointCloud>;
    using PointCloudConstPtr = std::shared_ptr<const PointCloud>;

    using PointRepresentation = pcl::PointRepresentation<PointT>;
    // using PointRepresentationPtr = std::shared_ptr<PointRepresentation>;
    using PointRepresentationConstPtr = std::shared_ptr<const PointRepresentation>;

    //  shared pointers
    typedef std::shared_ptr<KdTree<PointT>> Ptr;
    typedef std::shared_ptr<const KdTree<PointT>> ConstPtr;

    /** \brief Empty constructor for KdTree. Sets some internal values to their defaults.
     * \param[in] sorted set to true if the application that the tree will be used for requires sorted nearest neighbor
     * indices (default). False otherwise.
     */
    KdTree(bool sorted = true)
        : input_()
        , epsilon_(0.0f)
        , min_pts_(1)
        , sorted_(sorted)
        , point_representation_(new DefaultPointRepresentation<PointT>){};

    /** \brief Provide a pointer to the input dataset.
     * \param[in] cloud the const boost shared pointer to a PointCloud message
     * \param[in] indices the point indices subset that is to be used from \a cloud - if NULL the whole cloud is used
     */
    virtual void setInputCloud(const PointCloudConstPtr& cloud, const IndicesConstPtr& indices = IndicesConstPtr()) {
        input_ = cloud;
        indices_ = indices;
    }

    /** \brief Get a pointer to the vector of indices used. */
    inline IndicesConstPtr getIndices() const { return (indices_); }

    /** \brief Get a pointer to the input point cloud dataset. */
    inline PointCloudConstPtr getInputCloud() const { return (input_); }

    /** \brief Provide a pointer to the point representation to use to convert points into k-D vectors.
     * \param[in] point_representation the const boost shared pointer to a PointRepresentation
     */
    inline void setPointRepresentation(const PointRepresentationConstPtr& point_representation) {
        point_representation_ = point_representation;
        if (!input_) return;
        setInputCloud(input_, indices_); // Makes sense in derived classes to reinitialize the tree
    }

    /** \brief Get a pointer to the point representation used when converting points into k-D vectors. */
    inline PointRepresentationConstPtr getPointRepresentation() const { return (point_representation_); }

    /** \brief Destructor for KdTree. Deletes all allocated data arrays and destroys the kd-tree structures. */
    virtual ~KdTree(){};

    /** \brief Search for k-nearest neighbors for the given query point.
     * \param[in] p_q the given query point
     * \param[in] k the number of neighbors to search for
     * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
     * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k
     * a priori!)
     * \return number of neighbors found
     */
    virtual int nearestKSearch(
        const PointT& p_q, int k, std::vector<uint32_t>& k_indices, std::vector<float>& k_sqr_distances) const = 0;

    /** \brief Search for k-nearest neighbors for the given query point.
     *
     * \attention This method does not do any bounds checking for the input index
     * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
     *
     * \param[in] cloud the point cloud data
     * \param[in] index a \a valid index in \a cloud representing a \a valid (i.e., finite) query point
     * \param[in] k the number of neighbors to search for
     * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
     * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k
     * a priori!)
     *
     * \return number of neighbors found
     *
     * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
     */
    virtual int nearestKSearch(const PointCloud& cloud, int index, int k, std::vector<uint32_t>& k_indices,
        std::vector<float>& k_sqr_distances) const {
        assert(index >= 0 && index < static_cast<uint32_t>(cloud.points.size()) &&
               "Out-of-bounds error in nearestKSearch!");
        return (nearestKSearch(cloud.points[index], k, k_indices, k_sqr_distances));
    }

    /** \brief Search for k-nearest neighbors for the given query point.
     * This method accepts a different template parameter for the point type.
     * \param[in] point the given query point
     * \param[in] k the number of neighbors to search for
     * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
     * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k
     * a priori!)
     * \return number of neighbors found
     */
    template <typename PointTDiff>
    inline int nearestKSearchT(
        const PointTDiff& point, int k, std::vector<uint32_t>& k_indices, std::vector<float>& k_sqr_distances) const {
        PointT p;
        copyPoint(point, p);
        return (nearestKSearch(p, k, k_indices, k_sqr_distances));
    }

    /** \brief Search for k-nearest neighbors for the given query point (zero-copy).
     *
     * \attention This method does not do any bounds checking for the input index
     * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
     *
     * \param[in] index a \a valid index representing a \a valid query point in the dataset given
     * by \a setInputCloud. If indices were given in setInputCloud, index will be the position in
     * the indices vector.
     *
     * \param[in] k the number of neighbors to search for
     * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
     * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k
     * a priori!)
     * \return number of neighbors found
     *
     * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
     */
    virtual int nearestKSearch(
        int index, int k, std::vector<uint32_t>& k_indices, std::vector<float>& k_sqr_distances) const {
        if (indices_ == nullptr) {
            assert(index >= 0 && index < static_cast<uint32_t>(input_->points.size()) &&
                   "Out-of-bounds error in nearestKSearch!");
            return (nearestKSearch(input_->points[index], k, k_indices, k_sqr_distances));
        }
        assert(
            index >= 0 && index < static_cast<uint32_t>(indices_->size()) && "Out-of-bounds error in nearestKSearch!");
        return (nearestKSearch(input_->points[(*indices_)[index]], k, k_indices, k_sqr_distances));
    }

    /** \brief Search for all the nearest neighbors of the query point in a given radius.
     * \param[in] p_q the given query point
     * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
     * \param[out] k_indices the resultant indices of the neighboring points
     * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
     * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
     * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
     * returned.
     * \return number of neighbors found in radius
     */
    virtual int radiusSearch(const PointT& p_q, double radius, std::vector<uint32_t>& k_indices,
        std::vector<float>& k_sqr_distances, unsigned int max_nn = 0) const = 0;

    /** \brief Search for all the nearest neighbors of the query point in a given radius.
     *
     * \attention This method does not do any bounds checking for the input index
     * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
     *
     * \param[in] cloud the point cloud data
     * \param[in] index a \a valid index in \a cloud representing a \a valid (i.e., finite) query point
     * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
     * \param[out] k_indices the resultant indices of the neighboring points
     * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
     * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
     * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
     * returned.
     * \return number of neighbors found in radius
     *
     * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
     */
    virtual int radiusSearch(const PointCloud& cloud, int index, double radius, std::vector<uint32_t>& k_indices,
        std::vector<float>& k_sqr_distances, unsigned int max_nn = 0) const {
        assert(
            index >= 0 && index < static_cast<uint32_t>(cloud.points.size()) && "Out-of-bounds error in radiusSearch!");
        return (radiusSearch(cloud.points[index], radius, k_indices, k_sqr_distances, max_nn));
    }

    /** \brief Search for all the nearest neighbors of the query point in a given radius.
     * \param[in] point the given query point
     * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
     * \param[out] k_indices the resultant indices of the neighboring points
     * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
     * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
     * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
     * returned.
     * \return number of neighbors found in radius
     */
    template <typename PointTDiff>
    inline int radiusSearchT(const PointTDiff& point, double radius, std::vector<uint32_t>& k_indices,
        std::vector<float>& k_sqr_distances, unsigned int max_nn = 0) const {
        PointT p;
        copyPoint(point, p);
        return (radiusSearch(p, radius, k_indices, k_sqr_distances, max_nn));
    }

    /** \brief Search for all the nearest neighbors of the query point in a given radius (zero-copy).
     *
     * \attention This method does not do any bounds checking for the input index
     * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
     *
     * \param[in] index a \a valid index representing a \a valid query point in the dataset given
     * by \a setInputCloud. If indices were given in setInputCloud, index will be the position in
     * the indices vector.
     *
     * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
     * \param[out] k_indices the resultant indices of the neighboring points
     * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
     * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
     * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
     * returned.
     * \return number of neighbors found in radius
     *
     * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
     */
    virtual int radiusSearch(int index, double radius, std::vector<uint32_t>& k_indices, std::vector<float>& k_sqr_distances,
        unsigned int max_nn = 0) const {
        if (indices_ == nullptr) {
            assert(index >= 0 && index < static_cast<uint32_t>(input_->points.size()) &&
                   "Out-of-bounds error in radiusSearch!");
            return (radiusSearch(input_->points[index], radius, k_indices, k_sqr_distances, max_nn));
        }
        assert(index >= 0 && index < static_cast<uint32_t>(indices_->size()) && "Out-of-bounds error in radiusSearch!");
        return (radiusSearch(input_->points[(*indices_)[index]], radius, k_indices, k_sqr_distances, max_nn));
    }

    /** \brief Set the search epsilon precision (error bound) for nearest neighbors searches.
     * \param[in] eps precision (error bound) for nearest neighbors searches
     */
    virtual inline void setEpsilon(float eps) { epsilon_ = eps; }

    /** \brief Get the search epsilon precision (error bound) for nearest neighbors searches. */
    inline float getEpsilon() const { return (epsilon_); }

    /** \brief Minimum allowed number of k nearest neighbors points that a viable result must contain.
     * \param[in] min_pts the minimum number of neighbors in a viable neighborhood
     */
    inline void setMinPts(int min_pts) { min_pts_ = min_pts; }

    /** \brief Get the minimum allowed number of k nearest neighbors points that a viable result must contain. */
    inline int getMinPts() const { return (min_pts_); }

protected:
    /** \brief The input point cloud dataset containing the points we need to use. */
    PointCloudConstPtr input_;

    /** \brief A pointer to the vector of point indices to use. */
    IndicesConstPtr indices_;

    /** \brief Epsilon precision (error bound) for nearest neighbors searches. */
    float epsilon_;

    /** \brief Minimum allowed number of k nearest neighbors points that a viable result must contain. */
    int min_pts_;

    /** \brief Return the radius search neighbours sorted **/
    bool sorted_;

    /** \brief For converting different point structures into k-dimensional vectors for nearest-neighbor search. */
    PointRepresentationConstPtr point_representation_;

    /** \brief Class getName method. */
    virtual std::string getName() const = 0;
};



// And this is the "dataset to kd-tree" adaptor class:
template <typename shvecPointT> struct PointCloudAdaptor {
    //typedef typename shvecPointT::element_type vecPointT;
    //typedef typename vecPointT::value_type PointT;
    //typedef typename PointT::value_t value_t;
    typedef typename shvecPointT::element_type PC_type;
    typedef typename PC_type::coord_t PointT;
    typedef typename PointT::value_t value_t;

    shvecPointT obj;

    /// The constructor that sets the data set source
    PointCloudAdaptor(const shvecPointT& obj_) : obj(obj_) {}

    //PointCloudAdaptor(PointCloudAdaptor<shvecPointT>&&) = delete;
    //PointCloudAdaptor(const PointCloudAdaptor<shvecPointT>&) = delete;

    /// CRTP helper method
    inline const shvecPointT& derived() const { return obj; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return derived()->points.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline value_t kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0)
            return derived()->points[idx].x;
        else if (dim == 1)
            return derived()->points[idx].y;
        else
            return derived()->points[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it
    //   again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX> bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

}; // end of PointCloudAdaptor



  /** \brief KdTreeFLANN is a generic type of 3D spatial locator using kD-tree structures. The class is making use of
    * the FLANN (Fast Library for Approximate Nearest Neighbor) project by Marius Muja and David Lowe.
    *
    * \author Radu B. Rusu, Marius Muja
    * \ingroup kdtree 
    */
  template <typename PointT>
  class KdTreeFLANN : public pcl::KdTree<PointT>
  {
    public:
      using KdTree<PointT>::input_;
      using KdTree<PointT>::indices_;
      using KdTree<PointT>::epsilon_;
      using KdTree<PointT>::sorted_;
      using KdTree<PointT>::point_representation_;
      using KdTree<PointT>::nearestKSearch;
      using KdTree<PointT>::radiusSearch;

      using PointCloud = typename KdTree<PointT>::PointCloud;
      using PointCloudConstPtr = typename KdTree<PointT>::PointCloudConstPtr;

      using IndicesPtr = std::shared_ptr<std::vector<int> >;
      using IndicesConstPtr = std::shared_ptr<const std::vector<uint32_t>>;
     
      // typedef PointCloudAdaptor<std::shared_ptr<std::vector<PointT, Eigen::aligned_allocator<PointT>>>> PC2KD;
      typedef PointCloudAdaptor<PointCloudConstPtr> PC2KD;
      typedef ::nanoflann::KDTreeSingleIndexAdaptor<::nanoflann::L2_Simple_Adaptor<typename PointT::value_t, PC2KD, double>, PC2KD,
          3 /* dim */> FLANNIndex;
          

      // Boost shared pointers
      using Ptr = std::shared_ptr<KdTreeFLANN<PointT> >;
      using ConstPtr = std::shared_ptr<const KdTreeFLANN<PointT> >;

      /** \brief Default Constructor for KdTreeFLANN.
        * \param[in] sorted set to true if the application that the tree will be used for requires sorted nearest neighbor indices (default). False otherwise. 
        *
        * By setting sorted to false, the \ref radiusSearch operations will be faster.
        */
      KdTreeFLANN (bool sorted = true);

      /** \brief Copy constructor
        * \param[in] k the tree to copy into this
        */
      KdTreeFLANN (const KdTreeFLANN<PointT> &k);

      /** \brief Copy operator
        * \param[in] k the tree to copy into this
        */ 
      inline KdTreeFLANN<PointT>&
      operator = (const KdTreeFLANN<PointT>& k)
      {
        KdTree<PointT>::operator=(k);
        flann_index_ = k.flann_index_;
        cloud_ = k.cloud_;
        index_mapping_ = k.index_mapping_;
        identity_mapping_ = k.identity_mapping_;
        dim_ = k.dim_;
        total_nr_points_ = k.total_nr_points_;
        param_k_ = k.param_k_;
        param_radius_ = k.param_radius_;
        return (*this);
      }

      /** \brief Set the search epsilon precision (error bound) for nearest neighbors searches.
        * \param[in] eps precision (error bound) for nearest neighbors searches
        */
      void
      setEpsilon (float eps) override;

      void 
      setSortedResults (bool sorted);
      
      inline Ptr makeShared () { return Ptr (new KdTreeFLANN<PointT> (*this)); } 

      /** \brief Destructor for KdTreeFLANN. 
        * Deletes all allocated data arrays and destroys the kd-tree structures. 
        */
      ~KdTreeFLANN ()
      {
        cleanup ();
      }

      /** \brief Provide a pointer to the input dataset.
        * \param[in] cloud the const boost shared pointer to a PointCloud message
        * \param[in] indices the point indices subset that is to be used from \a cloud - if NULL the whole cloud is used
        */
      void 
      setInputCloud(const PointCloudConstPtr& cloud, const IndicesConstPtr& indices = IndicesConstPtr()) override;

      /** \brief Search for k-nearest neighbors for the given query point.
        * 
        * \attention This method does not do any bounds checking for the input index
        * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        * 
        * \param[in] point a given \a valid (i.e., finite) query point
        * \param[in] k the number of neighbors to search for
        * \param[out] k_indices the resultant indices of the neighboring points (must be resized to \a k a priori!)
        * \param[out] k_sqr_distances the resultant squared distances to the neighboring points (must be resized to \a k 
        * a priori!)
        * \return number of neighbors found
        * 
        * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        */
      int 
      nearestKSearch (const PointT &point, int k, 
                      std::vector<uint32_t> &k_indices, std::vector<float> &k_sqr_distances) const override;

      /** \brief Search for all the nearest neighbors of the query point in a given radius.
        * 
        * \attention This method does not do any bounds checking for the input index
        * (i.e., index >= cloud.points.size () || index < 0), and assumes valid (i.e., finite) data.
        * 
        * \param[in] point a given \a valid (i.e., finite) query point
        * \param[in] radius the radius of the sphere bounding all of p_q's neighbors
        * \param[out] k_indices the resultant indices of the neighboring points
        * \param[out] k_sqr_distances the resultant squared distances to the neighboring points
        * \param[in] max_nn if given, bounds the maximum returned neighbors to this value. If \a max_nn is set to
        * 0 or to a number higher than the number of points in the input cloud, all neighbors in \a radius will be
        * returned.
        * \return number of neighbors found in radius
        *
        * \exception asserts in debug mode if the index is not between 0 and the maximum number of points
        */
      int 
      radiusSearch(const PointT &point, double radius, std::vector<uint32_t> &k_indices,
                    std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const override;

    private:
      /** \brief Internal cleanup method. */
      void 
      cleanup ();

      /** \brief Converts a PointCloud to the internal FLANN point array representation. Returns the number
        * of points.
        * \param cloud the PointCloud 
        */
      void 
      convertCloudToArray(const PointCloud &cloud);

      /** \brief Converts a PointCloud with a given set of indices to the internal FLANN point array
        * representation. Returns the number of points.
        * \param[in] cloud the PointCloud data
        * \param[in] indices the point cloud indices
       */
      void 
      convertCloudToArray(const PointCloud& cloud, const std::vector<uint32_t>& indices);

    private:
      /** \brief Class getName method. */
      std::string 
      getName () const override { return ("KdTreeFLANN"); }

      /** \brief A FLANN index object. */
      std::shared_ptr<FLANNIndex> flann_index_;

      PC2KD pc2kd;

      /** \brief Internal pointer to data. */
      std::shared_ptr<std::vector<PointT, Eigen::aligned_allocator<PointT>>> cloud_;
      
      /** \brief mapping between internal and external indices. */
      std::vector<int> index_mapping_;
      
      /** \brief whether the mapping between internal and external indices is identity */
      bool identity_mapping_;

      /** \brief Tree dimensionality (i.e. the number of dimensions per point). */
      int dim_;

      /** \brief The total size of the data (either equal to the number of points in the input cloud or to the number of indices - if passed). */
      int total_nr_points_;

      /** \brief The KdTree search parameters for K-nearest neighbors. */
      ::nanoflann::SearchParams param_k_;

      /** \brief The KdTree search parameters for radius search. */
      ::nanoflann::SearchParams param_radius_;
  };
}


// #include <pcl/kdtree/impl/kdtree_flann.hpp>


///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
pcl::KdTreeFLANN<PointT>::KdTreeFLANN(bool sorted)
    : pcl::KdTree<PointT>(sorted)
    , flann_index_(nullptr)
    , identity_mapping_(false)
    , dim_(0)
    , total_nr_points_(0)
    , param_k_(::nanoflann::SearchParams(-1, epsilon_))
    , param_radius_(::nanoflann::SearchParams(-1, epsilon_, sorted))
    , pc2kd(nullptr) {}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
pcl::KdTreeFLANN<PointT>::KdTreeFLANN(const KdTreeFLANN<PointT>& k)
    : pcl::KdTree<PointT>(false)
    , flann_index_(nullptr)
    , identity_mapping_(false)
    , dim_(0)
    , total_nr_points_(0)
    , param_k_(::nanoflann::SearchParams(-1, epsilon_))
    , param_radius_(::nanoflann::SearchParams(-1, epsilon_, false))
    , pc2kd(nullptr) {
    *this = k;
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void pcl::KdTreeFLANN<PointT>::setEpsilon(float eps) {
    epsilon_ = eps;
    param_k_ = ::nanoflann::SearchParams(-1, epsilon_);
    param_radius_ = ::nanoflann::SearchParams(-1, epsilon_, sorted_);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void pcl::KdTreeFLANN<PointT>::setSortedResults(bool sorted) {
    sorted_ = sorted;
    param_k_ = ::nanoflann::SearchParams(-1, epsilon_);
    param_radius_ = ::nanoflann::SearchParams(-1, epsilon_, sorted_);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void pcl::KdTreeFLANN<PointT>::setInputCloud(const PointCloudConstPtr& cloud, const IndicesConstPtr& indices) {
    cleanup(); // Perform an automatic cleanup of structures

    epsilon_ = 0.0f;                                       // default error bound value
    dim_ = point_representation_->getNumberOfDimensions(); // Number of dimensions - default is 3 = xyz

    input_ = cloud;
    indices_ = indices;
    total_nr_points_ = cloud->points.size();

    // Allocate enough data
    //if (!input_) {
    //    vislib::sys::Log::DefaultLog.WriteError("[pcl::KdTreeFLANN::setInputCloud] Invalid input!\n");
    //    return;
    //}
    //if (indices != nullptr) {
    //    convertCloudToArray(*input_, *indices_);
    //} else {
    //    convertCloudToArray(*input_);
    //}
    //total_nr_points_ = static_cast<int>(index_mapping_.size());
    //if (total_nr_points_ == 0) {
    //    vislib::sys::Log::DefaultLog.WriteError("[pcl::KdTreeFLANN::setInputCloud] Cannot create a KDTree with an empty input cloud!\n");
    //    return;
    //}
    //
    // pc2kd = PC2KD(cloud_);


    pc2kd = PC2KD(cloud);
    flann_index_ = std::make_unique<FLANNIndex>(3 /*dim*/, pc2kd, ::nanoflann::KDTreeSingleIndexAdaptorParams(15 /* max leaf */));
    flann_index_->buildIndex();
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
int pcl::KdTreeFLANN<PointT>::nearestKSearch(
    const PointT& point, int k, std::vector<uint32_t>& k_indices, std::vector<float>& k_distances) const {
    assert(point_representation_->isValid(point) && "Invalid (NaN, Inf) point coordinates given to nearestKSearch!");

    // if (k > total_nr_points_) k = total_nr_points_;

    k_indices.resize(k);
    k_distances.resize(k);

    std::vector<float> query(dim_);
    point_representation_->vectorize(static_cast<PointT>(point), query);

    //::flann::Matrix<int> k_indices_mat(&k_indices[0], 1, k);
    //::flann::Matrix<float> k_distances_mat(&k_distances[0], 1, k);
    // Wrap the k_indices and k_distances vectors (no data copy)
    //const PC2KD pc2kd(cloud_);
    
    //flann_index_ = new FLANNIndex(3 /*dim*/, pc2kd, ::nanoflann::KDTreeSingleIndexAdaptorParams(10));
    //auto flann_idx = new FLANNIndex(3 /*dim*/, pc2kd, ::nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    //flann_index_.reset(        new FLANNIndex(3 /*dim*/, pc2kd, ::nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)));
    //flann_index_.reset(new FLANNIndex(3 /*dim*/, pc2kd, ::nanoflann::KDTreeSingleIndexAdaptorParams(10)));
    //flann_index_->buildIndex();
    //flann_index_->knnSearch(::flann::Matrix<float>(&query[0], 1, dim_), k_indices_mat, k_distances_mat, k, param_k_);

    nanoflann::KNNResultSet<float, uint32_t, int> resultSet(k);
    resultSet.init(k_indices.data(), k_distances.data());
    flann_index_->findNeighbors(resultSet, point.data, nanoflann::SearchParams(10));

    // Do mapping to original point cloud
    //if (!identity_mapping_) {
    //    for (size_t i = 0; i < static_cast<size_t>(k); ++i) {
    //        int& neighbor_index = k_indices[i];
    //        neighbor_index = index_mapping_[neighbor_index];
    //    }
    //}

    return (k);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
int pcl::KdTreeFLANN<PointT>::radiusSearch(const PointT& point, double radius, std::vector<uint32_t>& k_indices,
    std::vector<float>& k_sqr_dists, unsigned int max_nn) const {
    assert(point_representation_->isValid(point) && "Invalid (NaN, Inf) point coordinates given to radiusSearch!");

    std::vector<float> query(dim_);
    point_representation_->vectorize(static_cast<PointT>(point), query);

    // Has max_nn been set properly?
    if (max_nn == 0 || max_nn > static_cast<unsigned int>(total_nr_points_)) max_nn = total_nr_points_;

    std::vector<std::vector<int>> indices(1);
    std::vector<std::vector<float>> dists(1);

    //const PC2KD pc2kd(cloud_);
    //auto flann_idx = FLANNIndex(3 /*dim*/, pc2kd, ::nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    //*flann_index_ = flann_idx;
    //flann_index_ = new FLANNIndex(3 /*dim*/, pc2kd, ::nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    //flann_index_->buildIndex();

    std::vector<std::pair<size_t, double>> ret_matches;

    ::nanoflann::SearchParams params;
    //if (max_nn == static_cast<unsigned int>(total_nr_points_))
    //    params.max_neighbors = -1; // return all neighbors in radius
    //else
    //    params.max_neighbors = max_nn;

    
    //int neighbors_in_radius = flann_index_->radiusSearch(
    //    ::flann::Matrix<float>(&query[0], 1, dim_), indices, dists, static_cast<float>(radius * radius), params);

    int neighbors_in_radius = flann_index_->radiusSearch(point.data, radius, ret_matches, params);

    // XXX
    k_indices.clear();
    k_sqr_dists.clear();
    k_indices.reserve(ret_matches.size());
    k_sqr_dists.reserve(ret_matches.size());
    for (auto& element : ret_matches) {
        k_indices.push_back(element.first);
        k_sqr_dists.push_back(element.second);
    }
    
    // Do mapping to original point cloud
    //if (!identity_mapping_) {
    //    for (int i = 0; i < neighbors_in_radius; ++i) {
    //        uint32_t& neighbor_index = k_indices[i];
    //        neighbor_index = index_mapping_[neighbor_index];
    //    }
    //}

    return (neighbors_in_radius);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void pcl::KdTreeFLANN<PointT>::cleanup() {
    // Data array cleanup
    index_mapping_.clear();

    if (indices_) indices_.reset();
}

/////////////////////////////////////////////////////////////////////////////////////////////
//template <typename PointT>
//void pcl::KdTreeFLANN<PointT>::convertCloudToArray(const PointCloud& cloud) {
//    // No point in doing anything if the array is empty
//    if (cloud.points.empty()) {
//        cloud_.reset();
//        return;
//    }
//
//    int original_no_of_points = static_cast<int>(cloud.points.size());
//
//    cloud_.reset(new float[original_no_of_points * dim_]);
//    float* cloud_ptr = cloud_.get();
//    index_mapping_.reserve(original_no_of_points);
//    identity_mapping_ = true;
//
//    for (int cloud_index = 0; cloud_index < original_no_of_points; ++cloud_index) {
//        // Check if the point is invalid
//        if (!point_representation_->isValid(cloud.points[cloud_index])) {
//            identity_mapping_ = false;
//            continue;
//        }
//
//        index_mapping_.push_back(cloud_index);
//
//        point_representation_->vectorize(cloud.points[cloud_index], cloud_ptr);
//        cloud_ptr += dim_;
//    }
//}
//
/////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void pcl::KdTreeFLANN<PointT>::convertCloudToArray(const PointCloud& cloud, const std::vector<uint32_t>& indices) {
    // No point in doing anything if the array is empty
    if (cloud.points.empty()) {
        cloud_.reset();
        return;
    }

    int original_no_of_points = static_cast<int>(indices.size());

    cloud_.reset(new std::vector<PointT, Eigen::aligned_allocator<PointT>>(original_no_of_points));
    float* cloud_ptr = reinterpret_cast<float*>(cloud_.get()->data());
    index_mapping_.reserve(original_no_of_points);
    // its a subcloud -> false
    // true only identity:
    //     - indices size equals cloud size
    //     - indices only contain values between 0 and cloud.size - 1
    //     - no index is multiple times in the list
    //     => index is complete
    // But we can not guarantee that => identity_mapping_ = false
    identity_mapping_ = false;

    for (const int& index : indices) {
        // Check if the point is invalid
        if (!point_representation_->isValid(cloud.points[index])) continue;

        // map from 0 - N -> indices [0] - indices [N]
        index_mapping_.push_back(index); // If the returned index should be for the indices vector

        point_representation_->vectorize(cloud.points[index], cloud_ptr);
        cloud_ptr += dim_;
    }
}

 template <typename PointT>
 void pcl::KdTreeFLANN<PointT>::convertCloudToArray(const PointCloud& cloud) {
    // No point in doing anything if the array is empty
    if (cloud.points.empty()) {
        cloud_.reset();
        return;
    }

     auto original_no_of_points = static_cast<int>(cloud.points.size());

    cloud_ = std::make_shared<std::vector<PointT, Eigen::aligned_allocator<PointT>>>(cloud.points);
 }

#define PCL_INSTANTIATE_KdTreeFLANN(T) template class PCL_EXPORTS pcl::KdTreeFLANN<T>;
