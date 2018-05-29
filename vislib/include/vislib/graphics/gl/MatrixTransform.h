/*
 * MatrixTransform.h
 *
 * Copyright (C) 2018 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_OPENGL_MATRIXTRANSFORM_H_INCLUDED
#define VISLIB_OPENGL_MATRIXTRANSFORM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/math/Matrix.h"
#include "vislib/math/Quaternion.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * Provides exchange and modification of view and projection matrices
     * without falling back on graphics api functions.
     */
    class MatrixTransform  {
    public:

        // Type(s)
        typedef vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> MatrixType;


        /** Ctor. */
        MatrixTransform();

        /** Dtor. */
        virtual ~MatrixTransform(void);


        /**
        * Set the view matrix.
        *
        * @param vm The view matrix.
        */
        inline void SetViewMatrix(float vm[16]) {
            this->viewMatrix = MatrixType(vm);
            this->isMVPset = false;
        }

        /**
        * Set the projection matrix.
        *
        * @param pm The view matrix.
        */
        inline void SetProjectionMatrix(float pm[16]) {
            this->projectionMatrix = MatrixType(pm);
            this->isMVPset = false;
        }


        /**
        * Scale the view matrix.
        *
        * @param x The scaling factor for x coordinate.
        * @param y The scaling factor for y coordinate.
        * @param z The scaling factor for z coordinate.
        */
        inline void Scale(float x, float y, float z) {
            MatrixType scaleMat;
            scaleMat.SetAt(0, 0, x);
            scaleMat.SetAt(1, 1, y);
            scaleMat.SetAt(2, 2, z);
            this->viewMatrix = this->viewMatrix * scaleMat;
            this->isMVPset = false;
        }

        /**
        * Scale the view matrix.
        *
        * @param xyz The scaling factor for all three coordinates.
        */
        inline void Scale(float xyz) {
            this->Scale(xyz, xyz, xyz);
        }

        /**
        * Translate the view matrix.
        *
        * @param x The translation factor for x coordinate.
        * @param y The translation factor for y coordinate.
        * @param z The translation factor for z coordinate.
        */
        inline void Translate(float x, float y, float z) {
            MatrixType translateMat;
            translateMat.SetAt(0, 3, x);
            translateMat.SetAt(1, 3, y);
            translateMat.SetAt(2, 3, z);
            this->viewMatrix = this->viewMatrix * translateMat;
            this->isMVPset = false;
        }

        /**
        * Translate the view matrix.
        *
        * @param xyz One translation factor for all three coordinates.
        */
        inline void Translate(float xyz) {
            this->Translate(xyz, xyz, xyz);
        }

        /**
        * Rotate view matrix.
        *
        * @param quat The quaternion representing the rotation.
        */
        inline void Rotate(vislib::math::Quaternion<float> q) {
            this->viewMatrix = this->viewMatrix * static_cast<MatrixType>(q);
            this->isMVPset = false;
        }

        /**
        * Rotate view matrix.
        *
        * @param x     The x component of the rotation axis.
        * @param y     The y component of the rotation axis.
        * @param z     The z component of the rotation axis.
        * @param angle The rotation angle in rad.
        */
        inline void Rotate(float x, float y, float z, float angle) {
            vislib::math::Quaternion<float> rotQ(angle, vislib::math::Vector<float, 3>(x, y, z));
            this->Rotate(rotQ);
        }

        /**
        * Answer the view matrix.
        *
        * @return The view matrix.
        */
        inline MatrixType MV(void) {
            return this->viewMatrix;
        }

        /**
        * Answer the inverted view matrix.
        *
        * @return The inverted view matrix.
        */
        inline MatrixType MVinv(void) {
            MatrixType mv = this->MV();
            mv.Invert();
            return mv;
        }

        /**
        * Answer the transposed view matrix.
        *
        * @return The transposed view matrix.
        */
        inline MatrixType MVtransp(void) {
            MatrixType mv = this->MV();
            mv.Transpose();
            return mv;
        }

        /**
        * Answer the view projection matrix.
        *
        * @return The view projection matrix.
        */
        inline MatrixType MVP(void) {
            if (!this->isMVPset) {
                this->viewProjMatrix = (this->projectionMatrix * this->viewMatrix);
                this->isMVPset = true;
            }
            return this->viewProjMatrix;
        }

        /**
        * Answer the inverted view projection matrix.
        *
        * @return The inverted view projection matrix.
        */
        inline MatrixType MVPinv(void) {
            MatrixType mvp = this->MVP();
            mvp.Invert();
            return mvp;
        }

        /**
        * Answer the transposed view projection matrix.
        *
        * @return The transposed view projection matrix.
        */
        inline MatrixType MVPtransp(void) {
            MatrixType mvp = this->MVP();
            mvp.Transpose();
            return mvp;
        }

    private:

        /**********************************************************************
        * variables
        **********************************************************************/

        /** The view matrix (MV). */
        MatrixType viewMatrix;

        /** The projection matrix (P). */
        MatrixType projectionMatrix;

        /** The view projection matrix (MVP). */
        MatrixType viewProjMatrix;

        /** Indicates whether MVP is updated or not. */
        bool isMVPset;
    };

} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#endif /* VISLIB_OPENGL_MATRIXTRANSFORM_H_INCLUDED */

