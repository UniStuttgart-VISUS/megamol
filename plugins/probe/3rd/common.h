/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010, Willow Garage, Inc.
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

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <array>
#include <memory>
#include <vector>
#include "vislib/sys/Log.h"

#ifdef _DEBUG
#    define PCL_DEBUG vislib::sys::Log::DefaultLog.WriteWarn
#else
// most hacky solution just to make michael more happy :,(
namespace {
inline auto do_nothing = [](auto... xs) {};
}
#    define PCL_DEBUG do_nothing
#endif

namespace pcl {

using Indices = std::vector<uint32_t>;
using IndicesPtr = std::shared_ptr<Indices>;
using IndicesConstPtr = std::shared_ptr<const Indices>;

struct Vertices {
    Vertices() {}

    std::array<uint32_t, 3> vertices;

public:
    using Ptr = std::shared_ptr<Vertices>;
    using ConstPtr = std::shared_ptr<const Vertices>;
}; // struct Vertices


struct _PointXYZ {
    union EIGEN_ALIGN16 {
        float data[4];
        struct {
            float x;
            float y;
            float z;
        };
    };

    // inline pcl::Vector3fMap getVector3fMap() { return (pcl::Vector3fMap(data)); }
    // inline pcl::Vector3fMapConst getVector3fMap() const { return (pcl::Vector3fMapConst(data)); }
    // inline pcl::Vector4fMap getVector4fMap() { return (pcl::Vector4fMap(data)); }
    // inline pcl::Vector4fMapConst getVector4fMap() const { return (pcl::Vector4fMapConst(data)); }
    // inline pcl::Array3fMap getArray3fMap() { return (pcl::Array3fMap(data)); }
    // inline pcl::Array3fMapConst getArray3fMap() const { return (pcl::Array3fMapConst(data)); }
    // inline pcl::Array4fMap getArray4fMap() { return (pcl::Array4fMap(data)); }
    // inline pcl::Array4fMapConst getArray4fMap() const { return (pcl::Array4fMapConst(data)); }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/** \brief A point structure representing Euclidean xyz coordinates. (SSE friendly)
 * \ingroup common
 */
struct EIGEN_ALIGN16 PointXYZ : public _PointXYZ {
    inline PointXYZ(const _PointXYZ& p) {
        x = p.x;
        y = p.y;
        z = p.z;
        data[3] = 1.0f;
    }

    inline PointXYZ() {
        x = y = z = 0.0f;
        data[3] = 1.0f;
    }

    inline PointXYZ(float _x, float _y, float _z) {
        x = _x;
        y = _y;
        z = _z;
        data[3] = 1.0f;
    }

    inline PointXYZ operator-(const PointXYZ& rhs) const {
        return PointXYZ(this->x - rhs.x, this->y - rhs.y, this->z - rhs.z);
    }

    typedef float value_t;

    inline bool isfinite() const {
        return (std::isfinite(x) && std::isfinite(y) && std::isfinite(z));
    }

    friend std::ostream& operator<<(std::ostream& os, const PointXYZ& p);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct EIGEN_ALIGN16 _PointNormal {
    // This adds the members x,y,z which can also be accessed using the point (which is float[4])
    union EIGEN_ALIGN16 {
        float data[4];
        struct {
            float x;
            float y;
            float z;
        };
    };
    // This adds the member normal[3] which can also be accessed using the point (which is float[4])
    union EIGEN_ALIGN16 {
        float data_n[4];
        float normal[3];
        struct {
            float normal_x;
            float normal_y;
            float normal_z;
        };
    };
    union {
        struct {
            float curvature;
        };
        float data_c[4];
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/** \brief A point structure representing Euclidean xyz coordinates, together with normal coordinates and the surface
 * curvature estimate. (SSE friendly) \ingroup common
 */
struct PointNormal : public _PointNormal {
    inline PointNormal(const _PointNormal& p) {
        x = p.x;
        y = p.y;
        z = p.z;
        data[3] = 1.0f;
        normal_x = p.normal_x;
        normal_y = p.normal_y;
        normal_z = p.normal_z;
        data_n[3] = 0.0f;
        curvature = p.curvature;
    }

    inline PointNormal() {
        x = y = z = 0.0f;
        data[3] = 1.0f;
        normal_x = normal_y = normal_z = data_n[3] = 0.0f;
        curvature = 0.f;
    }

    inline bool isfinite() const {
        return (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(normal_x) &&
                std::isfinite(normal_y) && std::isfinite(normal_z));
    }
    typedef float value_t;
};


struct PCLHeader {
    PCLHeader() : seq(0), stamp() {}

    /** \brief Sequence number */
    uint32_t seq;
    /** \brief A timestamp associated with the time when the data was acquired
     *
     * The value represents microseconds since 1970-01-01 00:00:00 (the UNIX epoch).
     */
    uint64_t stamp;
    /** \brief Coordinate frame ID */
    std::string frame_id;

    using Ptr = std::shared_ptr<PCLHeader>;
    using ConstPtr = std::shared_ptr<const PCLHeader>;
}; // struct PCLHeader


struct PointIndices {
    PointIndices() = default;

    PCLHeader header;

    std::vector<uint32_t> indices;

public:
    using Ptr = std::shared_ptr<PointIndices>;
    using ConstPtr = std::shared_ptr<const PointIndices>;
}; // struct PointIndices


///////////////////////////////////////////////////////////////////////////////////
template <typename PointT> class PointCloud {
public:
    PointCloud() = default;
    ~PointCloud() = default;
    /** \brief The point data. */
    std::vector<PointT, Eigen::aligned_allocator<PointT>> points;
    using Ptr = std::shared_ptr<PointCloud<PointT>>;
    using ConstPtr = std::shared_ptr<const PointCloud<PointT>>;

    typedef PointT coord_t;

    /** \brief The point cloud header. It contains information about the acquisition time. */
    PCLHeader header;
    /** \brief The point cloud width (if organized as an image-structure). */
    uint32_t width;
    /** \brief The point cloud height (if organized as an image-structure). */
    uint32_t height;
    /** \brief True if no points are invalid (e.g., have NaN or Inf values in any of their floating point fields). */
    bool is_dense;
};

/////////////////////////////////////////////////////////////////////////////////////////
/** \brief PCL base class. Implements methods that are used by most PCL algorithms.
 * \ingroup common
 */
template <typename PointT> class PCLBase {
public:
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;

    using PointIndicesPtr = std::shared_ptr<PointIndices>;
    using PointIndicesConstPtr = std::shared_ptr<PointIndices const>;

    /** \brief Empty constructor. */
    PCLBase();

    /** \brief Copy constructor. */
    PCLBase(const PCLBase& base);

protected:
    /** \brief Destructor. */
    ~PCLBase() {
        input_.reset();
        indices_.reset();
    }

    /** \brief The input point cloud dataset. */
    PointCloudConstPtr input_;

    /** \brief A pointer to the vector of point indices to use. */
    IndicesPtr indices_;

    /** \brief Set to true if point indices are used. */
    bool use_indices_;

    /** \brief If no set of indices are given, we construct a set of fake indices that mimic the input PointCloud. */
    bool fake_indices_;

    /** \brief This method should get called before starting the actual computation.
     *
     * Internally, initCompute() does the following:
     *   - checks if an input dataset is given, and returns false otherwise
     *   - checks whether a set of input indices has been given. Returns true if yes.
     *   - if no input indices have been given, a fake set is created, which will be used until:
     *     - either a new set is given via setIndices(), or
     *     - a new cloud is given that has a different set of points. This will trigger an update on the set of fake
     * indices
     */
    bool initCompute();

    /** \brief This method should get called after finishing the actual computation.
     */
    bool deinitCompute();

    /** \brief Provide a pointer to the input dataset
     * \param[in] cloud the const shared pointer to a PointCloud message
     */
    virtual void setInputCloud(const PointCloudConstPtr& cloud);

    /** \brief Get a pointer to the input point cloud dataset. */
    inline PointCloudConstPtr const getInputCloud() const { return (input_); }

    /** \brief Provide a pointer to the vector of indices that represents the input data.
     * \param[in] indices a pointer to the indices that represent the input data.
     */
    virtual void setIndices(const IndicesPtr& indices);

    /** \brief Provide a pointer to the vector of indices that represents the input data.
     * \param[in] indices a pointer to the indices that represent the input data.
     */
    virtual void setIndices(const IndicesConstPtr& indices);

    /** \brief Provide a pointer to the vector of indices that represents the input data.
     * \param[in] indices a pointer to the indices that represent the input data.
     */
    virtual void setIndices(const PointIndicesConstPtr& indices);

    /** \brief Set the indices for the points laying within an interest region of
     * the point cloud.
     * \note you shouldn't call this method on unorganized point clouds!
     * \param[in] row_start the offset on rows
     * \param[in] col_start the offset on columns
     * \param[in] nb_rows the number of rows to be considered row_start included
     * \param[in] nb_cols the number of columns to be considered col_start included
     */
    virtual void setIndices(size_t row_start, size_t col_start, size_t nb_rows, size_t nb_cols);

    /** \brief Get a pointer to the vector of indices used. */
    inline IndicesPtr const getIndices() { return (indices_); }

    /** \brief Get a pointer to the vector of indices used. */
    inline IndicesConstPtr const getIndices() const { return (indices_); }

    /** \brief Override PointCloud operator[] to shorten code
     * \note this method can be called instead of (*input_)[(*indices_)[pos]]
     * or input_->points[(*indices_)[pos]]
     * \param[in] pos position in indices_ vector
     */
    inline const PointT& operator[](size_t pos) const { return ((*input_)[(*indices_)[pos]]); }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//////////////////////////////////////////////////////////////////////////////////////////////
/** \brief Compute the radius of a circumscribed circle for a triangle formed of three points pa, pb, and pc
 * \param pa the first point
 * \param pb the second point
 * \param pc the third point
 * \return the radius of the circumscribed circle
 * \ingroup common
 */
template <typename PointT> inline double getCircumcircleRadius(const PointT& pa, const PointT& pb, const PointT& pc);


template <typename PointT> inline double getCircumcircleRadius(const PointT& pa, const PointT& pb, const PointT& pc) {
    Eigen::Vector4f p1(pa.x, pa.y, pa.z, 0);
    Eigen::Vector4f p2(pb.x, pb.y, pb.z, 0);
    Eigen::Vector4f p3(pc.x, pc.y, pc.z, 0);

    double p2p1 = (p2 - p1).norm(), p3p2 = (p3 - p2).norm(), p1p3 = (p1 - p3).norm();
    // Calculate the area of the triangle using Heron's formula
    // (http://en.wikipedia.org/wiki/Heron's_formula)
    double semiperimeter = (p2p1 + p3p2 + p1p3) / 2.0;
    double area = sqrt(semiperimeter * (semiperimeter - p2p1) * (semiperimeter - p3p2) * (semiperimeter - p1p3));
    // Compute the radius of the circumscribed circle
    return ((p2p1 * p3p2 * p1p3) / (4.0 * area));
}


//////////////////////////////////////////////////////////////////////////////////////////////
/** \brief Extract the indices of a given point cloud as a new point cloud
 * \param[in] cloud_in the input point cloud dataset
 * \param[in] indices the vector of indices representing the points to be copied from \a cloud_in
 * \param[out] cloud_out the resultant output point cloud dataset
 * \note Assumes unique indices.
 * \ingroup common
 */
template <typename PointT>
inline void copyPointCloud(
    const pcl::PointCloud<PointT>& cloud_in, const std::vector<uint32_t>& indices, pcl::PointCloud<PointT>& cloud_out);

template <typename PointT>
void copyPointCloud(
    const pcl::PointCloud<PointT>& cloud_in, const std::vector<uint32_t>& indices, pcl::PointCloud<PointT>& cloud_out) {
    // Do we want to copy everything?
    if (indices.size() == cloud_in.points.size()) {
        cloud_out = cloud_in;
        return;
    }

    // Allocate enough space and copy the basics
    cloud_out.points.resize(indices.size());
    cloud_out.header = cloud_in.header;
    cloud_out.width = static_cast<uint32_t>(indices.size());
    cloud_out.height = 1;
    cloud_out.is_dense = cloud_in.is_dense;
    // cloud_out.sensor_orientation_ = cloud_in.sensor_orientation_;
    // cloud_out.sensor_origin_ = cloud_in.sensor_origin_;

    // Iterate over each point
    for (size_t i = 0; i < indices.size(); ++i) cloud_out.points[i] = cloud_in.points[indices[i]];
}

//////////////////////////////////////////////////////////////////////////////////////////////

/** \brief @b PointRepresentation provides a set of methods for converting a point structs/object into an
 * n-dimensional vector.
 * \note This is an abstract class.  Subclasses must set nr_dimensions_ to the appropriate value in the constructor
 * and provide an implementation of the pure virtual copyToFloatArray method.
 * \author Michael Dixon
 */
template <typename PointT> class PointRepresentation {
protected:
    /** \brief The number of dimensions in this point's vector (i.e. the "k" in "k-D") */
    int nr_dimensions_;
    /** \brief A vector containing the rescale factor to apply to each dimension. */
    std::vector<float> alpha_;
    /** \brief Indicates whether this point representation is trivial. It is trivial if and only if the following
     *  conditions hold:
     *  - the relevant data consists only of float values
     *  - the vectorize operation directly copies the first nr_dimensions_ elements of PointT to the out array
     *  - sizeof(PointT) is a multiple of sizeof(float)
     *  In short, a trivial point representation converts the input point to a float array that is the same as if
     *  the point was reinterpret_casted to a float array of length nr_dimensions_ . This value says that this
     *  representation can be trivial; it is only trivial if setRescaleValues() has not been set.
     */
    bool trivial_;

public:
    using Ptr = std::shared_ptr<PointRepresentation<PointT>>;
    using ConstPtr = std::shared_ptr<const PointRepresentation<PointT>>;

    /** \brief Empty constructor */
    PointRepresentation() : nr_dimensions_(0), alpha_(0), trivial_(false) {}

    /** \brief Empty destructor */
    virtual ~PointRepresentation() {}

    /** \brief Copy point data from input point to a float array. This method must be overridden in all subclasses.
     *  \param[in] p The input point
     *  \param[out] out A pointer to a float array.
     */
    virtual void copyToFloatArray(const PointT& p, float* out) const = 0;

    /** \brief Returns whether this point representation is trivial. It is trivial if and only if the following
     *  conditions hold:
     *  - the relevant data consists only of float values
     *  - the vectorize operation directly copies the first nr_dimensions_ elements of PointT to the out array
     *  - sizeof(PointT) is a multiple of sizeof(float)
     *  In short, a trivial point representation converts the input point to a float array that is the same as if
     *  the point was reinterpret_casted to a float array of length nr_dimensions_ . */
    inline bool isTrivial() const { return trivial_ && alpha_.empty(); }

    /** \brief Verify that the input point is valid.
     *  \param p The point to validate
     */
    virtual bool isValid(const PointT& p) const {
        bool is_valid = true;

        if (trivial_) {
            const float* temp = reinterpret_cast<const float*>(&p);

            for (int i = 0; i < nr_dimensions_; ++i) {
                if (!std::isfinite(temp[i])) {
                    is_valid = false;
                    break;
                }
            }
        } else {
            float* temp = new float[nr_dimensions_];
            copyToFloatArray(p, temp);

            for (int i = 0; i < nr_dimensions_; ++i) {
                if (!std::isfinite(temp[i])) {
                    is_valid = false;
                    break;
                }
            }
            delete[] temp;
        }
        return (is_valid);
    }

    /** \brief Convert input point into a vector representation, rescaling by \a alpha.
     * \param[in] p the input point
     * \param[out] out The output vector.  Can be of any type that implements the [] operator.
     */
    template <typename OutputType> void vectorize(const PointT& p, OutputType& out) const {
        float* temp = new float[nr_dimensions_];
        copyToFloatArray(p, temp);
        if (alpha_.empty()) {
            for (int i = 0; i < nr_dimensions_; ++i) out[i] = temp[i];
        } else {
            for (int i = 0; i < nr_dimensions_; ++i) out[i] = temp[i] * alpha_[i];
        }
        delete[] temp;
    }

    /** \brief Set the rescale values to use when vectorizing points
     * \param[in] rescale_array The array/vector of rescale values.  Can be of any type that implements the [] operator.
     */
    void setRescaleValues(const float* rescale_array) {
        alpha_.resize(nr_dimensions_);
        for (int i = 0; i < nr_dimensions_; ++i) alpha_[i] = rescale_array[i];
    }

    /** \brief Return the number of dimensions in the point's vector representation. */
    inline int getNumberOfDimensions() const { return (nr_dimensions_); }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/** \brief @b DefaultPointRepresentation extends PointRepresentation to define default behavior for common point types.
 */
template <typename PointDefault> class DefaultPointRepresentation : public PointRepresentation<PointDefault> {
    using PointRepresentation<PointDefault>::nr_dimensions_;
    using PointRepresentation<PointDefault>::trivial_;

public:
    // Boost shared pointers
    using Ptr = std::shared_ptr<DefaultPointRepresentation<PointDefault>>;
    using ConstPtr = std::shared_ptr<const DefaultPointRepresentation<PointDefault>>;

    DefaultPointRepresentation() {
        // If point type is unknown, assume it's a struct/array of floats, and compute the number of dimensions
        nr_dimensions_ = sizeof(PointDefault) / sizeof(float);
        // Limit the default representation to the first 3 elements
        if (nr_dimensions_ > 3) nr_dimensions_ = 3;

        trivial_ = true;
    }

    ~DefaultPointRepresentation() {}

    inline Ptr makeShared() const { return (Ptr(new DefaultPointRepresentation<PointDefault>(*this))); }

    void copyToFloatArray(const PointDefault& p, float* out) const override {
        // If point type is unknown, treat it as a struct/array of floats
        const float* ptr = reinterpret_cast<const float*>(&p);
        for (int i = 0; i < nr_dimensions_; ++i) out[i] = ptr[i];
    }
};


template <typename PointT> pcl::PCLBase<PointT>::PCLBase() : input_(), use_indices_(false), fake_indices_(false) {}

template <typename PointT>
pcl::PCLBase<PointT>::PCLBase(const PCLBase& base)
    : input_(base.input_)
    , indices_(base.indices_)
    , use_indices_(base.use_indices_)
    , fake_indices_(base.fake_indices_) {}

template <typename PointT> void pcl::PCLBase<PointT>::setInputCloud(const PointCloudConstPtr& cloud) { input_ = cloud; }


template <typename PointT> void pcl::PCLBase<PointT>::setIndices(const IndicesPtr& indices) {
    indices_ = indices;
    fake_indices_ = false;
    use_indices_ = true;
}

template <typename PointT> void pcl::PCLBase<PointT>::setIndices(const IndicesConstPtr& indices) {
    indices_.reset(new std::vector<uint32_t>(*indices));
    fake_indices_ = false;
    use_indices_ = true;
}


template <typename PointT> void pcl::PCLBase<PointT>::setIndices(const PointIndicesConstPtr& indices) {
    indices_.reset(new std::vector<uint32_t>(indices->indices));
    fake_indices_ = false;
    use_indices_ = true;
}


template <typename PointT>
void pcl::PCLBase<PointT>::setIndices(size_t row_start, size_t col_start, size_t nb_rows, size_t nb_cols) {
    if ((nb_rows > input_->height) || (row_start > input_->height)) {
        vislib::sys::Log::DefaultLog.WriteError("[PCLBase::setIndices] cloud is only %d height", input_->height);
        return;
    }

    if ((nb_cols > input_->width) || (col_start > input_->width)) {
        vislib::sys::Log::DefaultLog.WriteError("[PCLBase::setIndices] cloud is only %d width", input_->width);
        return;
    }

    size_t row_end = row_start + nb_rows;
    if (row_end > input_->height) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[PCLBase::setIndices] %d is out of rows range %d", row_end, input_->height);
        return;
    }

    size_t col_end = col_start + nb_cols;
    if (col_end > input_->width) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[PCLBase::setIndices] %d is out of columns range %d", col_end, input_->width);
        return;
    }

    indices_.reset(new std::vector<uint32_t>);
    indices_->reserve(nb_cols * nb_rows);
    for (size_t i = row_start; i < row_end; i++)
        for (size_t j = col_start; j < col_end; j++)
            indices_->push_back(static_cast<uint32_t>((i * input_->width) + j));
    fake_indices_ = false;
    use_indices_ = true;
}

template <typename PointT> bool pcl::PCLBase<PointT>::initCompute() {
    // Check if input was set
    if (!input_) return (false);

    // If no point indices have been given, construct a set of indices for the entire input point cloud
    if (!indices_) {
        fake_indices_ = true;
        indices_.reset(new std::vector<uint32_t>);
    }

    // If we have a set of fake indices, but they do not match the number of points in the cloud, update them
    if (fake_indices_ && indices_->size() != input_->points.size()) {
        size_t indices_size = indices_->size();
        try {
            indices_->resize(input_->points.size());
        } catch (const std::bad_alloc&) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[initCompute] Failed to allocate %lu indices.\n", input_->points.size());
        }
        for (size_t i = indices_size; i < indices_->size(); ++i) {
            (*indices_)[i] = static_cast<int>(i);
        }
    }

    return (true);
}

template <typename PointT> bool pcl::PCLBase<PointT>::deinitCompute() { return (true); }

#define PCL_INSTANTIATE_PCLBase(T) template class PCL_EXPORTS pcl::PCLBase<T>;


} // namespace pcl
