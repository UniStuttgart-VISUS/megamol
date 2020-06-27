/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *  Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
 *  Copyright (C) 2009 Hauke Heibel <hauke.heibel@gmail.com>
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
 * $Id$
 *
 */

#pragma once

#include "vislib/sys/Log.h"
#include <string>
#include "common.h"


#ifndef NOMINMAX
#define NOMINMAX
#endif

#if defined __GNUC__
#  pragma GCC system_header
#elif defined __SUNPRO_CC
#  pragma disable_warn
#endif

#include <cmath>
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace pcl
{
  /** \brief Compute the roots of a quadratic polynom x^2 + b*x + c = 0
    * \param[in] b linear parameter
    * \param[in] c constant parameter
    * \param[out] roots solutions of x^2 + b*x + c = 0
    */
  template <typename Scalar, typename Roots> void
  computeRoots2 (const Scalar &b, const Scalar &c, Roots &roots);

  /** \brief computes the roots of the characteristic polynomial of the input matrix m, which are the eigenvalues
    * \param[in] m input matrix
    * \param[out] roots roots of the characteristic polynomial of the input matrix m, which are the eigenvalues
    */
  template <typename Matrix, typename Roots> void
  computeRoots (const Matrix &m, Roots &roots);

  /** \brief determine the smallest eigenvalue and its corresponding eigenvector
    * \param[in] mat input matrix that needs to be symmetric and positive semi definite
    * \param[out] eigenvalue the smallest eigenvalue of the input matrix
    * \param[out] eigenvector the corresponding eigenvector to the smallest eigenvalue of the input matrix
    * \ingroup common
    */
  template <typename Matrix, typename Vector> void
  eigen22 (const Matrix &mat, typename Matrix::Scalar &eigenvalue, Vector &eigenvector);

  /** \brief determine the smallest eigenvalue and its corresponding eigenvector
    * \param[in] mat input matrix that needs to be symmetric and positive semi definite
    * \param[out] eigenvectors the corresponding eigenvector to the smallest eigenvalue of the input matrix
    * \param[out] eigenvalues the smallest eigenvalue of the input matrix
    * \ingroup common
    */
  template <typename Matrix, typename Vector> void
  eigen22 (const Matrix &mat, Matrix &eigenvectors, Vector &eigenvalues);

  /** \brief determines the corresponding eigenvector to the given eigenvalue of the symmetric positive semi definite input matrix
    * \param[in] mat symmetric positive semi definite input matrix
    * \param[in] eigenvalue the eigenvalue which corresponding eigenvector is to be computed
    * \param[out] eigenvector the corresponding eigenvector for the input eigenvalue
    * \ingroup common
    */
  template <typename Matrix, typename Vector> void
  computeCorrespondingEigenVector (const Matrix &mat, const typename Matrix::Scalar &eigenvalue, Vector &eigenvector);
  
  /** \brief determines the eigenvector and eigenvalue of the smallest eigenvalue of the symmetric positive semi definite input matrix
    * \param[in] mat symmetric positive semi definite input matrix
    * \param[out] eigenvalue smallest eigenvalue of the input matrix
    * \param[out] eigenvector the corresponding eigenvector for the input eigenvalue
    * \note if the smallest eigenvalue is not unique, this function may return any eigenvector that is consistent to the eigenvalue.
    * \ingroup common
    */
  template <typename Matrix, typename Vector> void
  eigen33 (const Matrix &mat, typename Matrix::Scalar &eigenvalue, Vector &eigenvector);

  /** \brief determines the eigenvalues of the symmetric positive semi definite input matrix
    * \param[in] mat symmetric positive semi definite input matrix
    * \param[out] evals resulting eigenvalues in ascending order
    * \ingroup common
    */
  template <typename Matrix, typename Vector> void
  eigen33 (const Matrix &mat, Vector &evals);

  /** \brief determines the eigenvalues and corresponding eigenvectors of the symmetric positive semi definite input matrix
    * \param[in] mat symmetric positive semi definite input matrix
    * \param[out] evecs corresponding eigenvectors in correct order according to eigenvalues
    * \param[out] evals resulting eigenvalues in ascending order
    * \ingroup common
    */
  template <typename Matrix, typename Vector> void
  eigen33 (const Matrix &mat, Matrix &evecs, Vector &evals);

  /** \brief Calculate the inverse of a 2x2 matrix
    * \param[in] matrix matrix to be inverted
    * \param[out] inverse the resultant inverted matrix
    * \note only the upper triangular part is taken into account => non symmetric matrices will give wrong results
    * \return determinant of the original matrix => if 0 no inverse exists => result is invalid
    * \ingroup common
    */
  template <typename Matrix> typename Matrix::Scalar
  invert2x2 (const Matrix &matrix, Matrix &inverse);

  /** \brief Calculate the inverse of a 3x3 symmetric matrix.
    * \param[in] matrix matrix to be inverted
    * \param[out] inverse the resultant inverted matrix
    * \note only the upper triangular part is taken into account => non symmetric matrices will give wrong results
    * \return determinant of the original matrix => if 0 no inverse exists => result is invalid
    * \ingroup common
    */
  template <typename Matrix> typename Matrix::Scalar
  invert3x3SymMatrix (const Matrix &matrix, Matrix &inverse);

  /** \brief Calculate the inverse of a general 3x3 matrix.
    * \param[in] matrix matrix to be inverted
    * \param[out] inverse the resultant inverted matrix
    * \return determinant of the original matrix => if 0 no inverse exists => result is invalid
    * \ingroup common
    */
  template <typename Matrix> typename Matrix::Scalar
  invert3x3Matrix (const Matrix &matrix, Matrix &inverse);

  /** \brief Calculate the determinant of a 3x3 matrix.
    * \param[in] matrix matrix
    * \return determinant of the matrix
    * \ingroup common
    */
  template <typename Matrix> typename Matrix::Scalar
  determinant3x3Matrix (const Matrix &matrix);
  
  /** \brief Get the unique 3D rotation that will rotate \a z_axis into (0,0,1) and \a y_direction into a vector
    * with x=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
    * \param[in] z_axis the z-axis
    * \param[in] y_direction the y direction
    * \param[out] transformation the resultant 3D rotation
    * \ingroup common
    */
  inline void
  getTransFromUnitVectorsZY (const Eigen::Vector3f& z_axis, 
                             const Eigen::Vector3f& y_direction,
                             Eigen::Affine3f& transformation);

  /** \brief Get the unique 3D rotation that will rotate \a z_axis into (0,0,1) and \a y_direction into a vector
    * with x=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
    * \param[in] z_axis the z-axis
    * \param[in] y_direction the y direction
    * \return the resultant 3D rotation
    * \ingroup common
    */
  inline Eigen::Affine3f
  getTransFromUnitVectorsZY (const Eigen::Vector3f& z_axis, 
                             const Eigen::Vector3f& y_direction);

  /** \brief Get the unique 3D rotation that will rotate \a x_axis into (1,0,0) and \a y_direction into a vector
    * with z=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
    * \param[in] x_axis the x-axis
    * \param[in] y_direction the y direction
    * \param[out] transformation the resultant 3D rotation
    * \ingroup common
    */
  inline void
  getTransFromUnitVectorsXY (const Eigen::Vector3f& x_axis, 
                             const Eigen::Vector3f& y_direction,
                             Eigen::Affine3f& transformation);

  /** \brief Get the unique 3D rotation that will rotate \a x_axis into (1,0,0) and \a y_direction into a vector
    * with z=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
    * \param[in] x_axis the x-axis
    * \param[in] y_direction the y direction
    * \return the resulting 3D rotation
    * \ingroup common
    */
  inline Eigen::Affine3f
  getTransFromUnitVectorsXY (const Eigen::Vector3f& x_axis, 
                             const Eigen::Vector3f& y_direction);

  /** \brief Get the unique 3D rotation that will rotate \a z_axis into (0,0,1) and \a y_direction into a vector
    * with x=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
    * \param[in] y_direction the y direction
    * \param[in] z_axis the z-axis
    * \param[out] transformation the resultant 3D rotation
    * \ingroup common
    */
  inline void
  getTransformationFromTwoUnitVectors (const Eigen::Vector3f& y_direction, 
                                       const Eigen::Vector3f& z_axis,
                                       Eigen::Affine3f& transformation);

  /** \brief Get the unique 3D rotation that will rotate \a z_axis into (0,0,1) and \a y_direction into a vector
    * with x=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
    * \param[in] y_direction the y direction
    * \param[in] z_axis the z-axis
    * \return transformation the resultant 3D rotation
    * \ingroup common
    */
  inline Eigen::Affine3f
  getTransformationFromTwoUnitVectors (const Eigen::Vector3f& y_direction,
                                       const Eigen::Vector3f& z_axis);

  /** \brief Get the transformation that will translate \a origin to (0,0,0) and rotate \a z_axis into (0,0,1)
    * and \a y_direction into a vector with x=0 (or into (0,1,0) should \a y_direction be orthogonal to \a z_axis)
    * \param[in] y_direction the y direction
    * \param[in] z_axis the z-axis
    * \param[in] origin the origin
    * \param[in] transformation the resultant transformation matrix
    * \ingroup common
    */
  inline void
  getTransformationFromTwoUnitVectorsAndOrigin (const Eigen::Vector3f& y_direction, 
                                                const Eigen::Vector3f& z_axis,
                                                const Eigen::Vector3f& origin, 
                                                Eigen::Affine3f& transformation);

  /** \brief Extract the Euler angles (XYZ-convention) from the given transformation
    * \param[in] t the input transformation matrix
    * \param[in] roll the resulting roll angle
    * \param[in] pitch the resulting pitch angle
    * \param[in] yaw the resulting yaw angle
    * \ingroup common
    */
  template <typename Scalar> void
  getEulerAngles (const Eigen::Transform<Scalar, 3, Eigen::Affine> &t, Scalar &roll, Scalar &pitch, Scalar &yaw);

  inline void
  getEulerAngles (const Eigen::Affine3f &t, float &roll, float &pitch, float &yaw)
  {
    getEulerAngles<float> (t, roll, pitch, yaw);
  }

  inline void
  getEulerAngles (const Eigen::Affine3d &t, double &roll, double &pitch, double &yaw)
  {
    getEulerAngles<double> (t, roll, pitch, yaw);
  }

  /** Extract x,y,z and the Euler angles (XYZ-convention) from the given transformation
    * \param[in] t the input transformation matrix
    * \param[out] x the resulting x translation
    * \param[out] y the resulting y translation
    * \param[out] z the resulting z translation
    * \param[out] roll the resulting roll angle
    * \param[out] pitch the resulting pitch angle
    * \param[out] yaw the resulting yaw angle
    * \ingroup common
    */
  template <typename Scalar> void
  getTranslationAndEulerAngles (const Eigen::Transform<Scalar, 3, Eigen::Affine> &t,
                                Scalar &x, Scalar &y, Scalar &z,
                                Scalar &roll, Scalar &pitch, Scalar &yaw);

  inline void
  getTranslationAndEulerAngles (const Eigen::Affine3f &t,
                                float &x, float &y, float &z,
                                float &roll, float &pitch, float &yaw)
  {
    getTranslationAndEulerAngles<float> (t, x, y, z, roll, pitch, yaw);
  }

  inline void
  getTranslationAndEulerAngles (const Eigen::Affine3d &t,
                                double &x, double &y, double &z,
                                double &roll, double &pitch, double &yaw)
  {
    getTranslationAndEulerAngles<double> (t, x, y, z, roll, pitch, yaw);
  }

  /** \brief Create a transformation from the given translation and Euler angles (XYZ-convention)
    * \param[in] x the input x translation
    * \param[in] y the input y translation
    * \param[in] z the input z translation
    * \param[in] roll the input roll angle
    * \param[in] pitch the input pitch angle
    * \param[in] yaw the input yaw angle
    * \param[out] t the resulting transformation matrix
    * \ingroup common
    */
  template <typename Scalar> void
  getTransformation (Scalar x, Scalar y, Scalar z, Scalar roll, Scalar pitch, Scalar yaw, 
                     Eigen::Transform<Scalar, 3, Eigen::Affine> &t);

  inline void
  getTransformation (float x, float y, float z, float roll, float pitch, float yaw, 
                     Eigen::Affine3f &t)
  {
    return (getTransformation<float> (x, y, z, roll, pitch, yaw, t));
  }

  inline void
  getTransformation (double x, double y, double z, double roll, double pitch, double yaw, 
                     Eigen::Affine3d &t)
  {
    return (getTransformation<double> (x, y, z, roll, pitch, yaw, t));
  }

  /** \brief Create a transformation from the given translation and Euler angles (XYZ-convention)
    * \param[in] x the input x translation
    * \param[in] y the input y translation
    * \param[in] z the input z translation
    * \param[in] roll the input roll angle
    * \param[in] pitch the input pitch angle
    * \param[in] yaw the input yaw angle
    * \return the resulting transformation matrix
    * \ingroup common
    */
  inline Eigen::Affine3f
  getTransformation (float x, float y, float z, float roll, float pitch, float yaw)
  {
    Eigen::Affine3f t;
    getTransformation<float> (x, y, z, roll, pitch, yaw, t);
    return (t);
  }

  /** \brief Write a matrix to an output stream
    * \param[in] matrix the matrix to output
    * \param[out] file the output stream
    * \ingroup common
    */
  template <typename Derived> void
  saveBinary (const Eigen::MatrixBase<Derived>& matrix, std::ostream& file);

  /** \brief Read a matrix from an input stream
    * \param[out] matrix the resulting matrix, read from the input stream
    * \param[in,out] file the input stream
    * \ingroup common
    */
  template <typename Derived> void
  loadBinary (Eigen::MatrixBase<Derived> const& matrix, std::istream& file);

// PCL_EIGEN_SIZE_MIN_PREFER_DYNAMIC gives the min between compile-time sizes. 0 has absolute priority, followed by 1,
// followed by Dynamic, followed by other finite values. The reason for giving Dynamic the priority over
// finite values is that min(3, Dynamic) should be Dynamic, since that could be anything between 0 and 3.
#define PCL_EIGEN_SIZE_MIN_PREFER_DYNAMIC(a,b) ((int (a) == 0 || int (b) == 0) ? 0 \
                           : (int (a) == 1 || int (b) == 1) ? 1 \
                           : (int (a) == Eigen::Dynamic || int (b) == Eigen::Dynamic) ? Eigen::Dynamic \
                           : (int (a) <= int (b)) ? int (a) : int (b))

  /** \brief Returns the transformation between two point sets. 
    * The algorithm is based on: 
    * "Least-squares estimation of transformation parameters between two point patterns",
    * Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    *
    * It estimates parameters \f$ c, \mathbf{R}, \f$ and \f$ \mathbf{t} \f$ such that
    * \f{align*}
    *   \frac{1}{n} \sum_{i=1}^n \vert\vert y_i - (c\mathbf{R}x_i + \mathbf{t}) \vert\vert_2^2
    * \f}
    * is minimized.
    *
    * The algorithm is based on the analysis of the covariance matrix
    * \f$ \Sigma_{\mathbf{x}\mathbf{y}} \in \mathbb{R}^{d \times d} \f$
    * of the input point sets \f$ \mathbf{x} \f$ and \f$ \mathbf{y} \f$ where
    * \f$d\f$ is corresponding to the dimension (which is typically small).
    * The analysis is involving the SVD having a complexity of \f$O(d^3)\f$
    * though the actual computational effort lies in the covariance
    * matrix computation which has an asymptotic lower bound of \f$O(dm)\f$ when
    * the input point sets have dimension \f$d \times m\f$.
    *
    * \param[in] src Source points \f$ \mathbf{x} = \left( x_1, \hdots, x_n \right) \f$
    * \param[in] dst Destination points \f$ \mathbf{y} = \left( y_1, \hdots, y_n \right) \f$.
    * \param[in] with_scaling Sets \f$ c=1 \f$ when <code>false</code> is passed. (default: false)
    * \return The homogeneous transformation 
    * \f{align*}
    *   T = \begin{bmatrix} c\mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix}
    * \f}
    * minimizing the resudiual above. This transformation is always returned as an
    * Eigen::Matrix.
    */
  template <typename Derived, typename OtherDerived> 
  typename Eigen::internal::umeyama_transform_matrix_type<Derived, OtherDerived>::type
  umeyama (const Eigen::MatrixBase<Derived>& src, const Eigen::MatrixBase<OtherDerived>& dst, bool with_scaling = false);

/** \brief Transform a point using an affine matrix
  * \param[in] point_in the vector to be transformed
  * \param[out] point_out the transformed vector
  * \param[in] transformation the transformation matrix
  *
  * \note Can be used with \c point_in = \c point_out
  */
  template<typename Scalar> inline void
  transformPoint (const Eigen::Matrix<Scalar, 3, 1> &point_in,
                        Eigen::Matrix<Scalar, 3, 1> &point_out,
                  const Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation)
  {
    Eigen::Matrix<Scalar, 4, 1> point;
    point << point_in, 1.0;
    point_out = (transformation * point).template head<3> ();
  }

  inline void
  transformPoint (const Eigen::Vector3f &point_in,
                        Eigen::Vector3f &point_out,
                  const Eigen::Affine3f &transformation)
  {
    transformPoint<float> (point_in, point_out, transformation);
  }

  inline void
  transformPoint (const Eigen::Vector3d &point_in,
                        Eigen::Vector3d &point_out,
                  const Eigen::Affine3d &transformation)
  {
    transformPoint<double> (point_in, point_out, transformation);
  }

/** \brief Transform a vector using an affine matrix
  * \param[in] vector_in the vector to be transformed
  * \param[out] vector_out the transformed vector
  * \param[in] transformation the transformation matrix
  *
  * \note Can be used with \c vector_in = \c vector_out
  */
  template <typename Scalar> inline void
  transformVector (const Eigen::Matrix<Scalar, 3, 1> &vector_in,
                         Eigen::Matrix<Scalar, 3, 1> &vector_out,
                   const Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation)
  {
    vector_out = transformation.linear () * vector_in;
  }

  inline void
  transformVector (const Eigen::Vector3f &vector_in,
                         Eigen::Vector3f &vector_out,
                   const Eigen::Affine3f &transformation)
  {
    transformVector<float> (vector_in, vector_out, transformation);
  }

  inline void
  transformVector (const Eigen::Vector3d &vector_in,
                         Eigen::Vector3d &vector_out,
                   const Eigen::Affine3d &transformation)
  {
    transformVector<double> (vector_in, vector_out, transformation);
  }

/** \brief Transform a line using an affine matrix
  * \param[in] line_in the line to be transformed
  * \param[out] line_out the transformed line
  * \param[in] transformation the transformation matrix
  *
  * Lines must be filled in this form:\n
  * line[0-2] = Origin coordinates of the vector\n
  * line[3-5] = Direction vector
  *
  * \note Can be used with \c line_in = \c line_out
  */
  template <typename Scalar> bool
  transformLine (const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &line_in,
                       Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &line_out,
                 const Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation);

  inline bool
  transformLine (const Eigen::VectorXf &line_in,
                       Eigen::VectorXf &line_out,
                 const Eigen::Affine3f &transformation)
  {
    return (transformLine<float> (line_in, line_out, transformation));
  }

  inline bool
  transformLine (const Eigen::VectorXd &line_in,
                       Eigen::VectorXd &line_out,
                 const Eigen::Affine3d &transformation)
  {
    return (transformLine<double> (line_in, line_out, transformation));
  }

/** \brief Transform plane vectors using an affine matrix
  * \param[in] plane_in the plane coefficients to be transformed
  * \param[out] plane_out the transformed plane coefficients to fill
  * \param[in] transformation the transformation matrix
  *
  * The plane vectors are filled in the form ax+by+cz+d=0
  * Can be used with non Hessian form planes coefficients
  * Can be used with \c plane_in = \c plane_out
  */
  template <typename Scalar> void
  transformPlane (const Eigen::Matrix<Scalar, 4, 1> &plane_in,
                        Eigen::Matrix<Scalar, 4, 1> &plane_out,
                  const Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation);

  inline void
  transformPlane (const Eigen::Matrix<double, 4, 1> &plane_in,
                        Eigen::Matrix<double, 4, 1> &plane_out,
                  const Eigen::Transform<double, 3, Eigen::Affine> &transformation)
  {
    transformPlane<double> (plane_in, plane_out, transformation);
  }

  inline void
  transformPlane (const Eigen::Matrix<float, 4, 1> &plane_in,
                        Eigen::Matrix<float, 4, 1> &plane_out,
                  const Eigen::Transform<float, 3, Eigen::Affine> &transformation)
  {
    transformPlane<float> (plane_in, plane_out, transformation);
  }

/** \brief Check coordinate system integrity
  * \param[in] line_x the first axis
  * \param[in] line_y the second axis
  * \param[in] norm_limit the limit to ignore norm rounding errors
  * \param[in] dot_limit the limit to ignore dot product rounding errors
  * \return True if the coordinate system is consistent, false otherwise.
  *
  * Lines must be filled in this form:\n
  * line[0-2] = Origin coordinates of the vector\n
  * line[3-5] = Direction vector
  *
  * Can be used like this :\n
  * line_x = X axis and line_y = Y axis\n
  * line_x = Z axis and line_y = X axis\n
  * line_x = Y axis and line_y = Z axis\n
  * Because X^Y = Z, Z^X = Y and Y^Z = X.
  * Do NOT invert line order !
  *
  * Determine whether a coordinate system is consistent or not by checking :\n
  * Line origins: They must be the same for the 2 lines\n
  * Norm: The 2 lines must be normalized\n
  * Dot products: Must be 0 or perpendicular vectors
  */
  template<typename Scalar> bool
  checkCoordinateSystem (const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &line_x,
                         const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &line_y,
                         const Scalar norm_limit = 1e-3,
                         const Scalar dot_limit = 1e-3);

  inline bool
  checkCoordinateSystem (const Eigen::Matrix<double, Eigen::Dynamic, 1> &line_x,
                         const Eigen::Matrix<double, Eigen::Dynamic, 1> &line_y,
                         const double norm_limit = 1e-3,
                         const double dot_limit = 1e-3)
  {
    return (checkCoordinateSystem<double> (line_x, line_y, norm_limit, dot_limit));
  }

  inline bool
  checkCoordinateSystem (const Eigen::Matrix<float, Eigen::Dynamic, 1> &line_x,
                         const Eigen::Matrix<float, Eigen::Dynamic, 1> &line_y,
                         const float norm_limit = 1e-3,
                         const float dot_limit = 1e-3)
  {
    return (checkCoordinateSystem<float> (line_x, line_y, norm_limit, dot_limit));
  }

/** \brief Check coordinate system integrity
  * \param[in] origin the origin of the coordinate system
  * \param[in] x_direction the first axis
  * \param[in] y_direction the second axis
  * \param[in] norm_limit the limit to ignore norm rounding errors
  * \param[in] dot_limit the limit to ignore dot product rounding errors
  * \return True if the coordinate system is consistent, false otherwise.
  *
  * Read the other variant for more information
  */
  template <typename Scalar> inline bool
  checkCoordinateSystem (const Eigen::Matrix<Scalar, 3, 1> &origin,
                         const Eigen::Matrix<Scalar, 3, 1> &x_direction,
                         const Eigen::Matrix<Scalar, 3, 1> &y_direction,
                         const Scalar norm_limit = 1e-3,
                         const Scalar dot_limit = 1e-3)
  {
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> line_x;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> line_y;
    line_x << origin, x_direction;
    line_y << origin, y_direction;
    return (checkCoordinateSystem<Scalar> (line_x, line_y, norm_limit, dot_limit));
  }

  inline bool
  checkCoordinateSystem (const Eigen::Matrix<double, 3, 1> &origin,
                         const Eigen::Matrix<double, 3, 1> &x_direction,
                         const Eigen::Matrix<double, 3, 1> &y_direction,
                         const double norm_limit = 1e-3,
                         const double dot_limit = 1e-3)
  {
    Eigen::Matrix<double, Eigen::Dynamic, 1> line_x;
    Eigen::Matrix<double, Eigen::Dynamic, 1> line_y;
    line_x.resize (6);
    line_y.resize (6);
    line_x << origin, x_direction;
    line_y << origin, y_direction;
    return (checkCoordinateSystem<double> (line_x, line_y, norm_limit, dot_limit));
  }

  inline bool
  checkCoordinateSystem (const Eigen::Matrix<float, 3, 1> &origin,
                         const Eigen::Matrix<float, 3, 1> &x_direction,
                         const Eigen::Matrix<float, 3, 1> &y_direction,
                         const float norm_limit = 1e-3,
                         const float dot_limit = 1e-3)
  {
    Eigen::Matrix<float, Eigen::Dynamic, 1> line_x;
    Eigen::Matrix<float, Eigen::Dynamic, 1> line_y;
    line_x.resize (6);
    line_y.resize (6);
    line_x << origin, x_direction;
    line_y << origin, y_direction;
    return (checkCoordinateSystem<float> (line_x, line_y, norm_limit, dot_limit));
  }

/** \brief Compute the transformation between two coordinate systems
  * \param[in] from_line_x X axis from the origin coordinate system
  * \param[in] from_line_y Y axis from the origin coordinate system
  * \param[in] to_line_x X axis from the destination coordinate system
  * \param[in] to_line_y Y axis from the destination coordinate system
  * \param[out] transformation the transformation matrix to fill
  * \return true if transformation was filled, false otherwise.
  *
  * Line must be filled in this form:\n
  * line[0-2] = Coordinate system origin coordinates \n
  * line[3-5] = Direction vector (norm doesn't matter)
  */
  template <typename Scalar> bool
  transformBetween2CoordinateSystems (const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> from_line_x,
                                      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> from_line_y,
                                      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> to_line_x,
                                      const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> to_line_y,
                                      Eigen::Transform<Scalar, 3, Eigen::Affine> &transformation);

  inline bool
  transformBetween2CoordinateSystems (const Eigen::Matrix<double, Eigen::Dynamic, 1> from_line_x,
                                      const Eigen::Matrix<double, Eigen::Dynamic, 1> from_line_y,
                                      const Eigen::Matrix<double, Eigen::Dynamic, 1> to_line_x,
                                      const Eigen::Matrix<double, Eigen::Dynamic, 1> to_line_y,
                                      Eigen::Transform<double, 3, Eigen::Affine> &transformation)
  {
    return (transformBetween2CoordinateSystems<double> (from_line_x, from_line_y, to_line_x, to_line_y, transformation));
  }

  inline bool
  transformBetween2CoordinateSystems (const Eigen::Matrix<float, Eigen::Dynamic, 1> from_line_x,
                                      const Eigen::Matrix<float, Eigen::Dynamic, 1> from_line_y,
                                      const Eigen::Matrix<float, Eigen::Dynamic, 1> to_line_x,
                                      const Eigen::Matrix<float, Eigen::Dynamic, 1> to_line_y,
                                      Eigen::Transform<float, 3, Eigen::Affine> &transformation)
  {
    return (transformBetween2CoordinateSystems<float> (from_line_x, from_line_y, to_line_x, to_line_y, transformation));
  }

}

// #include <pcl/common/impl/eigen.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar, typename Roots>
inline void pcl::computeRoots2(const Scalar& b, const Scalar& c, Roots& roots) {
    roots(0) = Scalar(0);
    Scalar d = Scalar(b * b - 4.0 * c);
    if (d < 0.0) // no real roots ! THIS SHOULD NOT HAPPEN!
        d = 0.0;

    Scalar sd = ::std::sqrt(d);

    roots(2) = 0.5f * (b + sd);
    roots(1) = 0.5f * (b - sd);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix, typename Roots> inline void pcl::computeRoots(const Matrix& m, Roots& roots) {
    using Scalar = typename Matrix::Scalar;

    // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
    // eigenvalues are the roots to this equation, all guaranteed to be
    // real-valued, because the matrix is symmetric.
    Scalar c0 = m(0, 0) * m(1, 1) * m(2, 2) + Scalar(2) * m(0, 1) * m(0, 2) * m(1, 2) - m(0, 0) * m(1, 2) * m(1, 2) -
                m(1, 1) * m(0, 2) * m(0, 2) - m(2, 2) * m(0, 1) * m(0, 1);
    Scalar c1 = m(0, 0) * m(1, 1) - m(0, 1) * m(0, 1) + m(0, 0) * m(2, 2) - m(0, 2) * m(0, 2) + m(1, 1) * m(2, 2) -
                m(1, 2) * m(1, 2);
    Scalar c2 = m(0, 0) + m(1, 1) + m(2, 2);

    if (std::abs(c0) < Eigen::NumTraits<Scalar>::epsilon()) // one root is 0 -> quadratic equation
        computeRoots2(c2, c1, roots);
    else {
        const Scalar s_inv3 = Scalar(1.0 / 3.0);
        const Scalar s_sqrt3 = std::sqrt(Scalar(3.0));
        // Construct the parameters used in classifying the roots of the equation
        // and in solving the equation for the roots in closed form.
        Scalar c2_over_3 = c2 * s_inv3;
        Scalar a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
        if (a_over_3 > Scalar(0)) a_over_3 = Scalar(0);

        Scalar half_b = Scalar(0.5) * (c0 + c2_over_3 * (Scalar(2) * c2_over_3 * c2_over_3 - c1));

        Scalar q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
        if (q > Scalar(0)) q = Scalar(0);

        // Compute the eigenvalues by solving for the roots of the polynomial.
        Scalar rho = std::sqrt(-a_over_3);
        Scalar theta = std::atan2(std::sqrt(-q), half_b) * s_inv3;
        Scalar cos_theta = std::cos(theta);
        Scalar sin_theta = std::sin(theta);
        roots(0) = c2_over_3 + Scalar(2) * rho * cos_theta;
        roots(1) = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
        roots(2) = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

        // Sort in increasing order.
        if (roots(0) >= roots(1)) std::swap(roots(0), roots(1));
        if (roots(1) >= roots(2)) {
            std::swap(roots(1), roots(2));
            if (roots(0) >= roots(1)) std::swap(roots(0), roots(1));
        }

        if (roots(0) <= 0) // eigenval for symmetric positive semi-definite matrix can not be negative! Set it to 0
            computeRoots2(c2, c1, roots);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix, typename Vector>
inline void pcl::eigen22(const Matrix& mat, typename Matrix::Scalar& eigenvalue, Vector& eigenvector) {
    // if diagonal matrix, the eigenvalues are the diagonal elements
    // and the eigenvectors are not unique, thus set to Identity
    if (std::abs(mat.coeff(1)) <= std::numeric_limits<typename Matrix::Scalar>::min()) {
        if (mat.coeff(0) < mat.coeff(2)) {
            eigenvalue = mat.coeff(0);
            eigenvector[0] = 1.0;
            eigenvector[1] = 0.0;
        } else {
            eigenvalue = mat.coeff(2);
            eigenvector[0] = 0.0;
            eigenvector[1] = 1.0;
        }
        return;
    }

    // 0.5 to optimize further calculations
    typename Matrix::Scalar trace = static_cast<typename Matrix::Scalar>(0.5) * (mat.coeff(0) + mat.coeff(3));
    typename Matrix::Scalar determinant = mat.coeff(0) * mat.coeff(3) - mat.coeff(1) * mat.coeff(1);

    typename Matrix::Scalar temp = trace * trace - determinant;

    if (temp < 0) temp = 0;

    eigenvalue = trace - ::std::sqrt(temp);

    eigenvector[0] = -mat.coeff(1);
    eigenvector[1] = mat.coeff(0) - eigenvalue;
    eigenvector.normalize();
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix, typename Vector>
inline void pcl::eigen22(const Matrix& mat, Matrix& eigenvectors, Vector& eigenvalues) {
    // if diagonal matrix, the eigenvalues are the diagonal elements
    // and the eigenvectors are not unique, thus set to Identity
    if (std::abs(mat.coeff(1)) <= std::numeric_limits<typename Matrix::Scalar>::min()) {
        if (mat.coeff(0) < mat.coeff(3)) {
            eigenvalues.coeffRef(0) = mat.coeff(0);
            eigenvalues.coeffRef(1) = mat.coeff(3);
            eigenvectors.coeffRef(0) = 1.0;
            eigenvectors.coeffRef(1) = 0.0;
            eigenvectors.coeffRef(2) = 0.0;
            eigenvectors.coeffRef(3) = 1.0;
        } else {
            eigenvalues.coeffRef(0) = mat.coeff(3);
            eigenvalues.coeffRef(1) = mat.coeff(0);
            eigenvectors.coeffRef(0) = 0.0;
            eigenvectors.coeffRef(1) = 1.0;
            eigenvectors.coeffRef(2) = 1.0;
            eigenvectors.coeffRef(3) = 0.0;
        }
        return;
    }

    // 0.5 to optimize further calculations
    typename Matrix::Scalar trace = static_cast<typename Matrix::Scalar>(0.5) * (mat.coeff(0) + mat.coeff(3));
    typename Matrix::Scalar determinant = mat.coeff(0) * mat.coeff(3) - mat.coeff(1) * mat.coeff(1);

    typename Matrix::Scalar temp = trace * trace - determinant;

    if (temp < 0)
        temp = 0;
    else
        temp = ::std::sqrt(temp);

    eigenvalues.coeffRef(0) = trace - temp;
    eigenvalues.coeffRef(1) = trace + temp;

    // either this is in a row or column depending on RowMajor or ColumnMajor
    eigenvectors.coeffRef(0) = -mat.coeff(1);
    eigenvectors.coeffRef(2) = mat.coeff(0) - eigenvalues.coeff(0);
    typename Matrix::Scalar norm =
        static_cast<typename Matrix::Scalar>(1.0) /
        static_cast<typename Matrix::Scalar>(::std::sqrt(
            eigenvectors.coeffRef(0) * eigenvectors.coeffRef(0) + eigenvectors.coeffRef(2) * eigenvectors.coeffRef(2)));
    eigenvectors.coeffRef(0) *= norm;
    eigenvectors.coeffRef(2) *= norm;
    eigenvectors.coeffRef(1) = eigenvectors.coeffRef(2);
    eigenvectors.coeffRef(3) = -eigenvectors.coeffRef(0);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix, typename Vector>
inline void pcl::computeCorrespondingEigenVector(
    const Matrix& mat, const typename Matrix::Scalar& eigenvalue, Vector& eigenvector) {
    using Scalar = typename Matrix::Scalar;
    // Scale the matrix so its entries are in [-1,1].  The scaling is applied
    // only when at least one matrix entry has magnitude larger than 1.

    Scalar scale = mat.cwiseAbs().maxCoeff();
    if (scale <= std::numeric_limits<Scalar>::min()) scale = Scalar(1.0);

    Matrix scaledMat = mat / scale;

    scaledMat.diagonal().array() -= eigenvalue / scale;

    Vector vec1 = scaledMat.row(0).cross(scaledMat.row(1));
    Vector vec2 = scaledMat.row(0).cross(scaledMat.row(2));
    Vector vec3 = scaledMat.row(1).cross(scaledMat.row(2));

    Scalar len1 = vec1.squaredNorm();
    Scalar len2 = vec2.squaredNorm();
    Scalar len3 = vec3.squaredNorm();

    if (len1 >= len2 && len1 >= len3)
        eigenvector = vec1 / std::sqrt(len1);
    else if (len2 >= len1 && len2 >= len3)
        eigenvector = vec2 / std::sqrt(len2);
    else
        eigenvector = vec3 / std::sqrt(len3);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix, typename Vector>
inline void pcl::eigen33(const Matrix& mat, typename Matrix::Scalar& eigenvalue, Vector& eigenvector) {
    using Scalar = typename Matrix::Scalar;
    // Scale the matrix so its entries are in [-1,1].  The scaling is applied
    // only when at least one matrix entry has magnitude larger than 1.

    Scalar scale = mat.cwiseAbs().maxCoeff();
    if (scale <= std::numeric_limits<Scalar>::min()) scale = Scalar(1.0);

    Matrix scaledMat = mat / scale;

    Vector eigenvalues;
    computeRoots(scaledMat, eigenvalues);

    eigenvalue = eigenvalues(0) * scale;

    scaledMat.diagonal().array() -= eigenvalues(0);

    Vector vec1 = scaledMat.row(0).cross(scaledMat.row(1));
    Vector vec2 = scaledMat.row(0).cross(scaledMat.row(2));
    Vector vec3 = scaledMat.row(1).cross(scaledMat.row(2));

    Scalar len1 = vec1.squaredNorm();
    Scalar len2 = vec2.squaredNorm();
    Scalar len3 = vec3.squaredNorm();

    if (len1 >= len2 && len1 >= len3)
        eigenvector = vec1 / std::sqrt(len1);
    else if (len2 >= len1 && len2 >= len3)
        eigenvector = vec2 / std::sqrt(len2);
    else
        eigenvector = vec3 / std::sqrt(len3);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix, typename Vector> inline void pcl::eigen33(const Matrix& mat, Vector& evals) {
    using Scalar = typename Matrix::Scalar;
    Scalar scale = mat.cwiseAbs().maxCoeff();
    if (scale <= std::numeric_limits<Scalar>::min()) scale = Scalar(1.0);

    Matrix scaledMat = mat / scale;
    computeRoots(scaledMat, evals);
    evals *= scale;
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix, typename Vector> inline void pcl::eigen33(const Matrix& mat, Matrix& evecs, Vector& evals) {
    using Scalar = typename Matrix::Scalar;
    // Scale the matrix so its entries are in [-1,1].  The scaling is applied
    // only when at least one matrix entry has magnitude larger than 1.

    Scalar scale = mat.cwiseAbs().maxCoeff();
    if (scale <= std::numeric_limits<Scalar>::min()) scale = Scalar(1.0);

    Matrix scaledMat = mat / scale;

    // Compute the eigenvalues
    computeRoots(scaledMat, evals);

    if ((evals(2) - evals(0)) <= Eigen::NumTraits<Scalar>::epsilon()) {
        // all three equal
        evecs.setIdentity();
    } else if ((evals(1) - evals(0)) <= Eigen::NumTraits<Scalar>::epsilon()) {
        // first and second equal
        Matrix tmp;
        tmp = scaledMat;
        tmp.diagonal().array() -= evals(2);

        Vector vec1 = tmp.row(0).cross(tmp.row(1));
        Vector vec2 = tmp.row(0).cross(tmp.row(2));
        Vector vec3 = tmp.row(1).cross(tmp.row(2));

        Scalar len1 = vec1.squaredNorm();
        Scalar len2 = vec2.squaredNorm();
        Scalar len3 = vec3.squaredNorm();

        if (len1 >= len2 && len1 >= len3)
            evecs.col(2) = vec1 / std::sqrt(len1);
        else if (len2 >= len1 && len2 >= len3)
            evecs.col(2) = vec2 / std::sqrt(len2);
        else
            evecs.col(2) = vec3 / std::sqrt(len3);

        evecs.col(1) = evecs.col(2).unitOrthogonal();
        evecs.col(0) = evecs.col(1).cross(evecs.col(2));
    } else if ((evals(2) - evals(1)) <= Eigen::NumTraits<Scalar>::epsilon()) {
        // second and third equal
        Matrix tmp;
        tmp = scaledMat;
        tmp.diagonal().array() -= evals(0);

        Vector vec1 = tmp.row(0).cross(tmp.row(1));
        Vector vec2 = tmp.row(0).cross(tmp.row(2));
        Vector vec3 = tmp.row(1).cross(tmp.row(2));

        Scalar len1 = vec1.squaredNorm();
        Scalar len2 = vec2.squaredNorm();
        Scalar len3 = vec3.squaredNorm();

        if (len1 >= len2 && len1 >= len3)
            evecs.col(0) = vec1 / std::sqrt(len1);
        else if (len2 >= len1 && len2 >= len3)
            evecs.col(0) = vec2 / std::sqrt(len2);
        else
            evecs.col(0) = vec3 / std::sqrt(len3);

        evecs.col(1) = evecs.col(0).unitOrthogonal();
        evecs.col(2) = evecs.col(0).cross(evecs.col(1));
    } else {
        Matrix tmp;
        tmp = scaledMat;
        tmp.diagonal().array() -= evals(2);

        Vector vec1 = tmp.row(0).cross(tmp.row(1));
        Vector vec2 = tmp.row(0).cross(tmp.row(2));
        Vector vec3 = tmp.row(1).cross(tmp.row(2));

        Scalar len1 = vec1.squaredNorm();
        Scalar len2 = vec2.squaredNorm();
        Scalar len3 = vec3.squaredNorm();
#    ifdef _WIN32
        Scalar* mmax = new Scalar[3];
#    else
        Scalar mmax[3];
#    endif
        unsigned int min_el = 2;
        unsigned int max_el = 2;
        if (len1 >= len2 && len1 >= len3) {
            mmax[2] = len1;
            evecs.col(2) = vec1 / std::sqrt(len1);
        } else if (len2 >= len1 && len2 >= len3) {
            mmax[2] = len2;
            evecs.col(2) = vec2 / std::sqrt(len2);
        } else {
            mmax[2] = len3;
            evecs.col(2) = vec3 / std::sqrt(len3);
        }

        tmp = scaledMat;
        tmp.diagonal().array() -= evals(1);

        vec1 = tmp.row(0).cross(tmp.row(1));
        vec2 = tmp.row(0).cross(tmp.row(2));
        vec3 = tmp.row(1).cross(tmp.row(2));

        len1 = vec1.squaredNorm();
        len2 = vec2.squaredNorm();
        len3 = vec3.squaredNorm();
        if (len1 >= len2 && len1 >= len3) {
            mmax[1] = len1;
            evecs.col(1) = vec1 / std::sqrt(len1);
            min_el = len1 <= mmax[min_el] ? 1 : min_el;
            max_el = len1 > mmax[max_el] ? 1 : max_el;
        } else if (len2 >= len1 && len2 >= len3) {
            mmax[1] = len2;
            evecs.col(1) = vec2 / std::sqrt(len2);
            min_el = len2 <= mmax[min_el] ? 1 : min_el;
            max_el = len2 > mmax[max_el] ? 1 : max_el;
        } else {
            mmax[1] = len3;
            evecs.col(1) = vec3 / std::sqrt(len3);
            min_el = len3 <= mmax[min_el] ? 1 : min_el;
            max_el = len3 > mmax[max_el] ? 1 : max_el;
        }

        tmp = scaledMat;
        tmp.diagonal().array() -= evals(0);

        vec1 = tmp.row(0).cross(tmp.row(1));
        vec2 = tmp.row(0).cross(tmp.row(2));
        vec3 = tmp.row(1).cross(tmp.row(2));

        len1 = vec1.squaredNorm();
        len2 = vec2.squaredNorm();
        len3 = vec3.squaredNorm();
        if (len1 >= len2 && len1 >= len3) {
            mmax[0] = len1;
            evecs.col(0) = vec1 / std::sqrt(len1);
            min_el = len3 <= mmax[min_el] ? 0 : min_el;
            max_el = len3 > mmax[max_el] ? 0 : max_el;
        } else if (len2 >= len1 && len2 >= len3) {
            mmax[0] = len2;
            evecs.col(0) = vec2 / std::sqrt(len2);
            min_el = len3 <= mmax[min_el] ? 0 : min_el;
            max_el = len3 > mmax[max_el] ? 0 : max_el;
        } else {
            mmax[0] = len3;
            evecs.col(0) = vec3 / std::sqrt(len3);
            min_el = len3 <= mmax[min_el] ? 0 : min_el;
            max_el = len3 > mmax[max_el] ? 0 : max_el;
        }

        unsigned mid_el = 3 - min_el - max_el;
        evecs.col(min_el) = evecs.col((min_el + 1) % 3).cross(evecs.col((min_el + 2) % 3)).normalized();
        evecs.col(mid_el) = evecs.col((mid_el + 1) % 3).cross(evecs.col((mid_el + 2) % 3)).normalized();
#    ifdef _WIN32
        delete[] mmax;
#    endif
    }
    // Rescale back to the original size.
    evals *= scale;
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix> inline typename Matrix::Scalar pcl::invert2x2(const Matrix& matrix, Matrix& inverse) {
    using Scalar = typename Matrix::Scalar;
    Scalar det = matrix.coeff(0) * matrix.coeff(3) - matrix.coeff(1) * matrix.coeff(2);

    if (det != 0) {
        // Scalar inv_det = Scalar (1.0) / det;
        inverse.coeffRef(0) = matrix.coeff(3);
        inverse.coeffRef(1) = -matrix.coeff(1);
        inverse.coeffRef(2) = -matrix.coeff(2);
        inverse.coeffRef(3) = matrix.coeff(0);
        inverse /= det;
    }
    return det;
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix>
inline typename Matrix::Scalar pcl::invert3x3SymMatrix(const Matrix& matrix, Matrix& inverse) {
    using Scalar = typename Matrix::Scalar;
    // elements
    // a b c
    // b d e
    // c e f
    //| a b c |-1             |   fd-ee    ce-bf   be-cd  |
    //| b d e |    =  1/det * |   ce-bf    af-cc   bc-ae  |
    //| c e f |               |   be-cd    bc-ae   ad-bb  |

    // det = a(fd-ee) + b(ec-fb) + c(eb-dc)

    Scalar fd_ee = matrix.coeff(4) * matrix.coeff(8) - matrix.coeff(7) * matrix.coeff(5);
    Scalar ce_bf = matrix.coeff(2) * matrix.coeff(5) - matrix.coeff(1) * matrix.coeff(8);
    Scalar be_cd = matrix.coeff(1) * matrix.coeff(5) - matrix.coeff(2) * matrix.coeff(4);

    Scalar det = matrix.coeff(0) * fd_ee + matrix.coeff(1) * ce_bf + matrix.coeff(2) * be_cd;

    if (det != 0) {
        // Scalar inv_det = Scalar (1.0) / det;
        inverse.coeffRef(0) = fd_ee;
        inverse.coeffRef(1) = inverse.coeffRef(3) = ce_bf;
        inverse.coeffRef(2) = inverse.coeffRef(6) = be_cd;
        inverse.coeffRef(4) = (matrix.coeff(0) * matrix.coeff(8) - matrix.coeff(2) * matrix.coeff(2));
        inverse.coeffRef(5) = inverse.coeffRef(7) =
            (matrix.coeff(1) * matrix.coeff(2) - matrix.coeff(0) * matrix.coeff(5));
        inverse.coeffRef(8) = (matrix.coeff(0) * matrix.coeff(4) - matrix.coeff(1) * matrix.coeff(1));
        inverse /= det;
    }
    return det;
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix> inline typename Matrix::Scalar pcl::invert3x3Matrix(const Matrix& matrix, Matrix& inverse) {
    using Scalar = typename Matrix::Scalar;

    //| a b c |-1             |   ie-hf    hc-ib   fb-ec  |
    //| d e f |    =  1/det * |   gf-id    ia-gc   dc-fa  |
    //| g h i |               |   hd-ge    gb-ha   ea-db  |
    // det = a(ie-hf) + d(hc-ib) + g(fb-ec)

    Scalar ie_hf = matrix.coeff(8) * matrix.coeff(4) - matrix.coeff(7) * matrix.coeff(5);
    Scalar hc_ib = matrix.coeff(7) * matrix.coeff(2) - matrix.coeff(8) * matrix.coeff(1);
    Scalar fb_ec = matrix.coeff(5) * matrix.coeff(1) - matrix.coeff(4) * matrix.coeff(2);
    Scalar det = matrix.coeff(0) * (ie_hf) + matrix.coeff(3) * (hc_ib) + matrix.coeff(6) * (fb_ec);

    if (det != 0) {
        inverse.coeffRef(0) = ie_hf;
        inverse.coeffRef(1) = hc_ib;
        inverse.coeffRef(2) = fb_ec;
        inverse.coeffRef(3) = matrix.coeff(6) * matrix.coeff(5) - matrix.coeff(8) * matrix.coeff(3);
        inverse.coeffRef(4) = matrix.coeff(8) * matrix.coeff(0) - matrix.coeff(6) * matrix.coeff(2);
        inverse.coeffRef(5) = matrix.coeff(3) * matrix.coeff(2) - matrix.coeff(5) * matrix.coeff(0);
        inverse.coeffRef(6) = matrix.coeff(7) * matrix.coeff(3) - matrix.coeff(6) * matrix.coeff(4);
        inverse.coeffRef(7) = matrix.coeff(6) * matrix.coeff(1) - matrix.coeff(7) * matrix.coeff(0);
        inverse.coeffRef(8) = matrix.coeff(4) * matrix.coeff(0) - matrix.coeff(3) * matrix.coeff(1);

        inverse /= det;
    }
    return det;
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Matrix> inline typename Matrix::Scalar pcl::determinant3x3Matrix(const Matrix& matrix) {
    // result is independent of Row/Col Major storage!
    return matrix.coeff(0) * (matrix.coeff(4) * matrix.coeff(8) - matrix.coeff(5) * matrix.coeff(7)) +
           matrix.coeff(1) * (matrix.coeff(5) * matrix.coeff(6) - matrix.coeff(3) * matrix.coeff(8)) +
           matrix.coeff(2) * (matrix.coeff(3) * matrix.coeff(7) - matrix.coeff(4) * matrix.coeff(6));
}

//////////////////////////////////////////////////////////////////////////////////////////
void pcl::getTransFromUnitVectorsZY(
    const Eigen::Vector3f& z_axis, const Eigen::Vector3f& y_direction, Eigen::Affine3f& transformation) {
    Eigen::Vector3f tmp0 = (y_direction.cross(z_axis)).normalized();
    Eigen::Vector3f tmp1 = (z_axis.cross(tmp0)).normalized();
    Eigen::Vector3f tmp2 = z_axis.normalized();

    transformation(0, 0) = tmp0[0];
    transformation(0, 1) = tmp0[1];
    transformation(0, 2) = tmp0[2];
    transformation(0, 3) = 0.0f;
    transformation(1, 0) = tmp1[0];
    transformation(1, 1) = tmp1[1];
    transformation(1, 2) = tmp1[2];
    transformation(1, 3) = 0.0f;
    transformation(2, 0) = tmp2[0];
    transformation(2, 1) = tmp2[1];
    transformation(2, 2) = tmp2[2];
    transformation(2, 3) = 0.0f;
    transformation(3, 0) = 0.0f;
    transformation(3, 1) = 0.0f;
    transformation(3, 2) = 0.0f;
    transformation(3, 3) = 1.0f;
}

//////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f pcl::getTransFromUnitVectorsZY(const Eigen::Vector3f& z_axis, const Eigen::Vector3f& y_direction) {
    Eigen::Affine3f transformation;
    getTransFromUnitVectorsZY(z_axis, y_direction, transformation);
    return (transformation);
}

//////////////////////////////////////////////////////////////////////////////////////////
void pcl::getTransFromUnitVectorsXY(
    const Eigen::Vector3f& x_axis, const Eigen::Vector3f& y_direction, Eigen::Affine3f& transformation) {
    Eigen::Vector3f tmp2 = (x_axis.cross(y_direction)).normalized();
    Eigen::Vector3f tmp1 = (tmp2.cross(x_axis)).normalized();
    Eigen::Vector3f tmp0 = x_axis.normalized();

    transformation(0, 0) = tmp0[0];
    transformation(0, 1) = tmp0[1];
    transformation(0, 2) = tmp0[2];
    transformation(0, 3) = 0.0f;
    transformation(1, 0) = tmp1[0];
    transformation(1, 1) = tmp1[1];
    transformation(1, 2) = tmp1[2];
    transformation(1, 3) = 0.0f;
    transformation(2, 0) = tmp2[0];
    transformation(2, 1) = tmp2[1];
    transformation(2, 2) = tmp2[2];
    transformation(2, 3) = 0.0f;
    transformation(3, 0) = 0.0f;
    transformation(3, 1) = 0.0f;
    transformation(3, 2) = 0.0f;
    transformation(3, 3) = 1.0f;
}

//////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f pcl::getTransFromUnitVectorsXY(const Eigen::Vector3f& x_axis, const Eigen::Vector3f& y_direction) {
    Eigen::Affine3f transformation;
    getTransFromUnitVectorsXY(x_axis, y_direction, transformation);
    return (transformation);
}

//////////////////////////////////////////////////////////////////////////////////////////
void pcl::getTransformationFromTwoUnitVectors(
    const Eigen::Vector3f& y_direction, const Eigen::Vector3f& z_axis, Eigen::Affine3f& transformation) {
    getTransFromUnitVectorsZY(z_axis, y_direction, transformation);
}

//////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f pcl::getTransformationFromTwoUnitVectors(
    const Eigen::Vector3f& y_direction, const Eigen::Vector3f& z_axis) {
    Eigen::Affine3f transformation;
    getTransformationFromTwoUnitVectors(y_direction, z_axis, transformation);
    return (transformation);
}

void pcl::getTransformationFromTwoUnitVectorsAndOrigin(const Eigen::Vector3f& y_direction,
    const Eigen::Vector3f& z_axis, const Eigen::Vector3f& origin, Eigen::Affine3f& transformation) {
    getTransformationFromTwoUnitVectors(y_direction, z_axis, transformation);
    Eigen::Vector3f translation = transformation * origin;
    transformation(0, 3) = -translation[0];
    transformation(1, 3) = -translation[1];
    transformation(2, 3) = -translation[2];
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar>
void pcl::getEulerAngles(
    const Eigen::Transform<Scalar, 3, Eigen::Affine>& t, Scalar& roll, Scalar& pitch, Scalar& yaw) {
    roll = std::atan2(t(2, 1), t(2, 2));
    pitch = asin(-t(2, 0));
    yaw = std::atan2(t(1, 0), t(0, 0));
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar>
void pcl::getTranslationAndEulerAngles(const Eigen::Transform<Scalar, 3, Eigen::Affine>& t, Scalar& x, Scalar& y,
    Scalar& z, Scalar& roll, Scalar& pitch, Scalar& yaw) {
    x = t(0, 3);
    y = t(1, 3);
    z = t(2, 3);
    roll = std::atan2(t(2, 1), t(2, 2));
    pitch = asin(-t(2, 0));
    yaw = std::atan2(t(1, 0), t(0, 0));
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar>
void pcl::getTransformation(Scalar x, Scalar y, Scalar z, Scalar roll, Scalar pitch, Scalar yaw,
    Eigen::Transform<Scalar, 3, Eigen::Affine>& t) {
    Scalar A = std::cos(yaw), B = sin(yaw), C = std::cos(pitch), D = sin(pitch), E = std::cos(roll), F = sin(roll),
           DE = D * E, DF = D * F;

    t(0, 0) = A * C;
    t(0, 1) = A * DF - B * E;
    t(0, 2) = B * F + A * DE;
    t(0, 3) = x;
    t(1, 0) = B * C;
    t(1, 1) = A * E + B * DF;
    t(1, 2) = B * DE - A * F;
    t(1, 3) = y;
    t(2, 0) = -D;
    t(2, 1) = C * F;
    t(2, 2) = C * E;
    t(2, 3) = z;
    t(3, 0) = 0;
    t(3, 1) = 0;
    t(3, 2) = 0;
    t(3, 3) = 1;
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Derived> void pcl::saveBinary(const Eigen::MatrixBase<Derived>& matrix, std::ostream& file) {
    uint32_t rows = static_cast<uint32_t>(matrix.rows()), cols = static_cast<uint32_t>(matrix.cols());
    file.write(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<char*>(&cols), sizeof(cols));
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) {
            typename Derived::Scalar tmp = matrix(i, j);
            file.write(reinterpret_cast<const char*>(&tmp), sizeof(tmp));
        }
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Derived> void pcl::loadBinary(Eigen::MatrixBase<Derived> const& matrix_, std::istream& file) {
    Eigen::MatrixBase<Derived>& matrix = const_cast<Eigen::MatrixBase<Derived>&>(matrix_);

    uint32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (matrix.rows() != static_cast<int>(rows) || matrix.cols() != static_cast<int>(cols))
        matrix.derived().resize(rows, cols);

    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) {
            typename Derived::Scalar tmp;
            file.read(reinterpret_cast<char*>(&tmp), sizeof(tmp));
            matrix(i, j) = tmp;
        }
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Derived, typename OtherDerived>
typename Eigen::internal::umeyama_transform_matrix_type<Derived, OtherDerived>::type pcl::umeyama(
    const Eigen::MatrixBase<Derived>& src, const Eigen::MatrixBase<OtherDerived>& dst, bool with_scaling) {
#    if EIGEN_VERSION_AT_LEAST(3, 3, 0)
    return Eigen::umeyama(src, dst, with_scaling);
#    else
    using TransformationMatrixType =
        typename Eigen::internal::umeyama_transform_matrix_type<Derived, OtherDerived>::type;
    using Scalar = typename Eigen::internal::traits<TransformationMatrixType>::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Index = typename Derived::Index;

    static_assert(!Eigen::NumTraits<Scalar>::IsComplex, "Numeric type must be real.");
    static_assert((Eigen::internal::is_same<Scalar, typename Eigen::internal::traits<OtherDerived>::Scalar>::value),
        "You mixed different numeric types. You need to use the cast method of matrixbase to cast numeric types "
        "explicitly.");

    enum { Dimension = PCL_EIGEN_SIZE_MIN_PREFER_DYNAMIC(Derived::RowsAtCompileTime, OtherDerived::RowsAtCompileTime) };

    using VectorType = Eigen::Matrix<Scalar, Dimension, 1>;
    using MatrixType = Eigen::Matrix<Scalar, Dimension, Dimension>;
    using RowMajorMatrixType = typename Eigen::internal::plain_matrix_type_row_major<Derived>::type;

    const Index m = src.rows(); // dimension
    const Index n = src.cols(); // number of measurements

    // required for demeaning ...
    const RealScalar one_over_n = 1 / static_cast<RealScalar>(n);

    // computation of mean
    const VectorType src_mean = src.rowwise().sum() * one_over_n;
    const VectorType dst_mean = dst.rowwise().sum() * one_over_n;

    // demeaning of src and dst points
    const RowMajorMatrixType src_demean = src.colwise() - src_mean;
    const RowMajorMatrixType dst_demean = dst.colwise() - dst_mean;

    // Eq. (36)-(37)
    const Scalar src_var = src_demean.rowwise().squaredNorm().sum() * one_over_n;

    // Eq. (38)
    const MatrixType sigma(one_over_n * dst_demean * src_demean.transpose());

    Eigen::JacobiSVD<MatrixType> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // Initialize the resulting transformation with an identity matrix...
    TransformationMatrixType Rt = TransformationMatrixType::Identity(m + 1, m + 1);

    // Eq. (39)
    VectorType S = VectorType::Ones(m);

    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) S(m - 1) = -1;

    // Eq. (40) and (43)
    Rt.block(0, 0, m, m).noalias() = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

    if (with_scaling) {
        // Eq. (42)
        const Scalar c = Scalar(1) / src_var * svd.singularValues().dot(S);

        // Eq. (41)
        Rt.col(m).head(m) = dst_mean;
        Rt.col(m).head(m).noalias() -= c * Rt.topLeftCorner(m, m) * src_mean;
        Rt.block(0, 0, m, m) *= c;
    } else {
        Rt.col(m).head(m) = dst_mean;
        Rt.col(m).head(m).noalias() -= Rt.topLeftCorner(m, m) * src_mean;
    }

    return (Rt);
#    endif
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar>
bool pcl::transformLine(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& line_in,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& line_out,
    const Eigen::Transform<Scalar, 3, Eigen::Affine>& transformation) {
    if (line_in.innerSize() != 6 || line_out.innerSize() != 6) {
        PCL_DEBUG("transformLine: lines size != 6\n");
        return (false);
    }

    Eigen::Matrix<Scalar, 3, 1> point, vector;
    point << line_in.template head<3>();
    vector << line_out.template tail<3>();

    pcl::transformPoint(point, point, transformation);
    pcl::transformVector(vector, vector, transformation);
    line_out << point, vector;
    return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar>
void pcl::transformPlane(const Eigen::Matrix<Scalar, 4, 1>& plane_in, Eigen::Matrix<Scalar, 4, 1>& plane_out,
    const Eigen::Transform<Scalar, 3, Eigen::Affine>& transformation) {
    Eigen::Hyperplane<Scalar, 3> plane;
    plane.coeffs() << plane_in;
    plane.transform(transformation);
    plane_out << plane.coeffs();

// Versions prior to 3.3.2 don't normalize the result
#    if !EIGEN_VERSION_AT_LEAST(3, 3, 2)
    plane_out /= plane_out.template head<3>().norm();
#    endif
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar>
bool pcl::checkCoordinateSystem(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& line_x,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& line_y, const Scalar norm_limit, const Scalar dot_limit) {
    if (line_x.innerSize() != 6 || line_y.innerSize() != 6) {
        PCL_DEBUG("checkCoordinateSystem: lines size != 6\n");
        return (false);
    }

    if (line_x.template head<3>() != line_y.template head<3>()) {
        PCL_DEBUG("checkCoorZdinateSystem: vector origins are different !\n");
        return (false);
    }

    // Make a copy of vector directions
    // X^Y = Z | Y^Z = X | Z^X = Y
    Eigen::Matrix<Scalar, 3, 1> v_line_x(line_x.template tail<3>()), v_line_y(line_y.template tail<3>()),
        v_line_z(v_line_x.cross(v_line_y));

    // Check vectors norms
    if (v_line_x.norm() < 1 - norm_limit || v_line_x.norm() > 1 + norm_limit) {
        PCL_DEBUG("checkCoordinateSystem: line_x norm %d != 1\n", v_line_x.norm());
        return (false);
    }

    if (v_line_y.norm() < 1 - norm_limit || v_line_y.norm() > 1 + norm_limit) {
        PCL_DEBUG("checkCoordinateSystem: line_y norm %d != 1\n", v_line_y.norm());
        return (false);
    }

    if (v_line_z.norm() < 1 - norm_limit || v_line_z.norm() > 1 + norm_limit) {
        PCL_DEBUG("checkCoordinateSystem: line_z norm %d != 1\n", v_line_z.norm());
        return (false);
    }

    // Check vectors perendicularity
    if (std::abs(v_line_x.dot(v_line_y)) > dot_limit) {
        PCL_DEBUG("checkCSAxis: line_x dot line_y %e =  > %e\n", v_line_x.dot(v_line_y), dot_limit);
        return (false);
    }

    if (std::abs(v_line_x.dot(v_line_z)) > dot_limit) {
        PCL_DEBUG("checkCSAxis: line_x dot line_z = %e > %e\n", v_line_x.dot(v_line_z), dot_limit);
        return (false);
    }

    if (std::abs(v_line_y.dot(v_line_z)) > dot_limit) {
        PCL_DEBUG("checkCSAxis: line_y dot line_z = %e > %e\n", v_line_y.dot(v_line_z), dot_limit);
        return (false);
    }

    return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////
template <typename Scalar>
bool pcl::transformBetween2CoordinateSystems(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> from_line_x,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> from_line_y,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> to_line_x, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> to_line_y,
    Eigen::Transform<Scalar, 3, Eigen::Affine>& transformation) {
    if (from_line_x.innerSize() != 6 || from_line_y.innerSize() != 6 || to_line_x.innerSize() != 6 ||
        to_line_y.innerSize() != 6) {
        PCL_DEBUG("transformBetween2CoordinateSystems: lines size != 6\n");
        return (false);
    }

    // Check if coordinate systems are valid
    if (!pcl::checkCoordinateSystem(from_line_x, from_line_y) || !pcl::checkCoordinateSystem(to_line_x, to_line_y)) {
        PCL_DEBUG("transformBetween2CoordinateSystems: coordinate systems invalid !\n");
        return (false);
    }

    // Convert lines into Vector3 :
    Eigen::Matrix<Scalar, 3, 1> fr0(from_line_x.template head<3>()),
        fr1(from_line_x.template head<3>() + from_line_x.template tail<3>()),
        fr2(from_line_y.template head<3>() + from_line_y.template tail<3>()),

        to0(to_line_x.template head<3>()), to1(to_line_x.template head<3>() + to_line_x.template tail<3>()),
        to2(to_line_y.template head<3>() + to_line_y.template tail<3>());

    // Code is inspired from http://stackoverflow.com/a/15277421/1816078
    // Define matrices and points :
    Eigen::Transform<Scalar, 3, Eigen::Affine> T2, T3 = Eigen::Transform<Scalar, 3, Eigen::Affine>::Identity();
    Eigen::Matrix<Scalar, 3, 1> x1, y1, z1, x2, y2, z2;

    // Axes of the coordinate system "fr"
    x1 = (fr1 - fr0).normalized(); // the versor (unitary vector) of the (fr1-fr0) axis vector
    y1 = (fr2 - fr0).normalized();

    // Axes of the coordinate system "to"
    x2 = (to1 - to0).normalized();
    y2 = (to2 - to0).normalized();

    // Transform from CS1 to CS2
    // Note: if fr0 == (0,0,0) --> CS1 == CS2 --> T2 = Identity
    T2.linear() << x1, y1, x1.cross(y1);

    // Transform from CS1 to CS3
    T3.linear() << x2, y2, x2.cross(y2);

    // Identity matrix = transform to CS2 to CS3
    // Note: if CS1 == CS2 --> transformation = T3
    transformation = Eigen::Transform<Scalar, 3, Eigen::Affine>::Identity();
    transformation.linear() = T3.linear() * T2.linear().inverse();
    transformation.translation() = to0 - (transformation.linear() * fr0);
    return (true);
}

