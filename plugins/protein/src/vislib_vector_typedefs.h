//
// vislib_vector_typedefs.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Feb 18, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINPLUGIN_VISLIB_VECTOR_TYPEDEFS_H_INCLUDED
#define MMPROTEINPLUGIN_VISLIB_VECTOR_TYPEDEFS_H_INCLUDED

#include "vislib/math/Vector.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/Cuboid.h"

// TODO rename to vislib_math_typedefs

namespace megamol {
namespace protein {

typedef vislib::math::Vector<char, 2> Vec2c;
typedef vislib::math::Vector<int, 2> Vec2i;
typedef vislib::math::Vector<unsigned int, 2> Vec2u;
typedef vislib::math::Vector<float, 2> Vec2f;
typedef vislib::math::Vector<double, 2> Vec2d;
typedef vislib::math::Vector<bool, 2> Vec2b;

typedef vislib::math::Vector<char, 3> Vec3c;
typedef vislib::math::Vector<int, 3> Vec3i;
typedef vislib::math::Vector<unsigned int, 3> Vec3u;
typedef vislib::math::Vector<float, 3> Vec3f;
typedef vislib::math::Vector<double, 3> Vec3d;
typedef vislib::math::Vector<bool, 3> Vec3b;

typedef vislib::math::Vector<char, 4> Vec4c;
typedef vislib::math::Vector<int, 4> Vec4i;
typedef vislib::math::Vector<unsigned int, 4> Vec4u;
typedef vislib::math::Vector<float, 4> Vec4f;
typedef vislib::math::Vector<double, 4> Vec4d;
typedef vislib::math::Vector<bool, 4> Vec4b;

typedef vislib::math::Matrix<char, 2, vislib::math::COLUMN_MAJOR> Mat2c;
typedef vislib::math::Matrix<int, 2, vislib::math::COLUMN_MAJOR> Mat2i;
typedef vislib::math::Matrix<unsigned int, 2, vislib::math::COLUMN_MAJOR> Mat2u;
typedef vislib::math::Matrix<float, 2, vislib::math::COLUMN_MAJOR> Mat2f;
typedef vislib::math::Matrix<double, 2, vislib::math::COLUMN_MAJOR> Mat2d;
typedef vislib::math::Matrix<bool, 2, vislib::math::COLUMN_MAJOR> Mat2b;

typedef vislib::math::Matrix<char, 3, vislib::math::COLUMN_MAJOR> Mat3c;
typedef vislib::math::Matrix<int, 3, vislib::math::COLUMN_MAJOR> Mat3i;
typedef vislib::math::Matrix<unsigned int, 3, vislib::math::COLUMN_MAJOR> Mat3u;
typedef vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> Mat3f;
typedef vislib::math::Matrix<double, 3, vislib::math::COLUMN_MAJOR> Mat3d;
typedef vislib::math::Matrix<bool, 3, vislib::math::COLUMN_MAJOR> Mat3b;

typedef vislib::math::Matrix<char, 4, vislib::math::COLUMN_MAJOR> Mat4c;
typedef vislib::math::Matrix<int, 4, vislib::math::COLUMN_MAJOR> Mat4i;
typedef vislib::math::Matrix<unsigned int, 4, vislib::math::COLUMN_MAJOR> Mat4u;
typedef vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> Mat4f;
typedef vislib::math::Matrix<double, 4, vislib::math::COLUMN_MAJOR> Mat4d;
typedef vislib::math::Matrix<bool, 4, vislib::math::COLUMN_MAJOR> Mat4b;

typedef vislib::math::Cuboid<unsigned int> Cubeu;
typedef vislib::math::Cuboid<int> Cubei;
typedef vislib::math::Cuboid<float> Cubef;

} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_VISLIB_VECTOR_TYPEDEFS_H_INCLUDED
