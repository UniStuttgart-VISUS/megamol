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
     * Provides exchange and modification of model view and projection matrices
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
        * Set the model view matrix.
        *
        * @param vm The model view matrix.
        */
        inline void SetModelViewMatrix(float vm[16]) {
            this->modelViewMatrix = MatrixType(vm);
            this->isMVPset = false;
        }

        /**
        * Set the projection matrix.
        *
        * @param pm The model view matrix.
        */
        inline void SetProjectionMatrix(float pm[16]) {
            this->projectionMatrix = MatrixType(pm);
            this->isMVPset = false;
        }


        /**
        * Scale the model view matrix.
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
            this->modelViewMatrix = this->modelViewMatrix * scaleMat;
            this->isMVPset = false;
        }

        /**
        * Scale the model view matrix.
        *
        * @param xyz The scaling factor for all three coordinates.
        */
        inline void Scale(float xyz) {
            this->Scale(xyz, xyz, xyz);
        }

        /**
        * Translate the model view matrix.
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
            this->modelViewMatrix = this->modelViewMatrix * translateMat;
            this->isMVPset = false;
        }

        /**
        * Translate the model view matrix.
        *
        * @param xyz The translation factor for all three coordinates.
        */
        inline void Translate(float xyz) {
            this->Translate(xyz, xyz, xyz);
        }

        /**
        * Rotate model view matrix.
        *
        * @param quat The ...
        */
        inline void Rotate(vislib::math::Quaternion<float> q) {
            this->modelViewMatrix = this->modelViewMatrix * Quat2RotMat(q);
            this->isMVPset = false;
        }

        /**
        * Rotate model view matrix.
        *
        * @param x     The ...
        * @param y     The ...
        * @param z     The ...
        * @param angle The ...
        */
        inline void Rotate(float x, float y, float z, float angle) {
            vislib::math::Quaternion<float> rotQ(angle, vislib::math::Vector<float, 3>(x, y, z));
            this->Rotate(rotQ);
        }


        /**
        * Answer the model view matrix.
        *
        * @return The model view matrix.
        */
        inline MatrixType MV(void) {
            return this->modelViewMatrix;
        }

        /**
        * Answer the inverted model view matrix.
        *
        * @return The inverted model view matrix.
        */
        inline MatrixType MVinv(void) {
            MatrixType mv = this->MV();
            mv.Invert();
            return mv;
        }

        /**
        * Answer the transposed model view matrix.
        *
        * @return The transposed model view matrix.
        */
        inline MatrixType MVtransp(void) {
            MatrixType mv = this->MV();
            mv.Transpose();
            return mv;
        }

        /**
        * Answer the model view projection matrix.
        *
        * @return The model view projection matrix.
        */
        inline MatrixType MVP(void) {
            if (!this->isMVPset) {
                this->modelViewProjMatrix = (this->projectionMatrix * this->modelViewMatrix);
                this->isMVPset = true;
            }
            return this->modelViewProjMatrix;
        }

        /**
        * Answer the inverted model view projection matrix.
        *
        * @return The inverted model view projection matrix.
        */
        inline MatrixType MVPinv(void) {
            MatrixType mvp = this->MVP();
            mvp.Invert();
            return mvp;
        }

        /**
        * Answer the transposed model view projection matrix.
        *
        * @return The transposed model view projection matrix.
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

        /** The model view matrix (MV). */
        MatrixType modelViewMatrix;
        /** The projection matrix (P). */
        MatrixType projectionMatrix;

        /** The model view projection matrix (MVP). */
        MatrixType modelViewProjMatrix;
        /** Indicates whether MVP is updated or not. */
        bool isMVPset;

        /**********************************************************************
        * functions
        **********************************************************************/

        /** Transform quaternion to rotation matrix.
        *
        * @param q The quaternion.
        *
        * @return The rotation matrix.
        */
        MatrixType Quat2RotMat(vislib::math::Quaternion<float> q) const;

    };

} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#endif /* VISLIB_OPENGL_MATRIXTRANSFORM_H_INCLUDED */

