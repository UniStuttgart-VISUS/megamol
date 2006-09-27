/*
 * Beholder.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_BEHOLDER_H_INCLUDED
#define VISLIB_BEHOLDER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/IllegalParamException.h"
#include "vislib/AbstractPoint3D.h"
#include "vislib/Point3D.h"
#include "vislib/AbstractVector3D.h"
#include "vislib/Vector3D.h"


namespace vislib {
namespace graphics {

    /**
     * class modelling a 3d scene beholder
     */
    template<class T > class Beholder {
    public:

        /**
         * default ctor
         *
         * Initializes the position to (0, 0, 0), lookAt to (0, 0, -1), 
         * up to (0, 1, 0), and the updateCounter to 0.
         */
        Beholder(void);

        /**
         * init ctor
         *
         * Initializes the view parameters as described below:
         *
         * The position of the beholder is set to position.
         *
         * The look at point of the beholder is set to lookAt, if the value of
         * lookAt is diffrent from the value of position. If the values are
         * equal, the look at point is set to position + (0, 0, -1).
         *
         * The up vector of the beholder is set to up, if up is not a null 
         * vector and if up is not parallel to the vector (position - lookAt).
         * Otherwise the up vector is set to (0, 1, 0). If this vector would be
         * parallel to the vector (position - lookAt), the up vector will be 
         * set to (0, 0, 1).
         *
         * The updateCounter will be set to 1 if all parameters were valid and
         * the member were set to these values. The value of updateCounter will
         * be set to 0 if at least one parameter value was invalid and a 
         * default value was used, as described above.
         *
         * Remark: The orthonormal coordinate system of the beholde is also 
         * updated which need multiple vector operations, including expensive 
         * vector normalisation.
         *
         * @param position The position for the beholder in world coordinates.
         * @param lookAt The look at point for the beholder in world 
         *               coordinates
         * @param up The up vector for the beholder
         */
        template <class Sp1, class Sp2, class Sp3> 
        Beholder(const math::AbstractPoint3D<T, Sp1> &position, 
            const math::AbstractPoint3D<T, Sp2> &lookAt, 
            const math::AbstractVector3D<T, Sp3> &up);

        /**
         * copy ctor
         *
         * copies all values from rhs but updateCounter which is set to 0.
         *
         * @param rhs The object to be cloned.
         */
        Beholder(const Beholder<T> &rhs);

        /**
         * assignment operator
         *
         * copies all values from rhs but updateCounter which is set to 0.
         *
         * @param rhs The object to be cloned.
         *
         * @return reference to this
         */
        Beholder& operator=(const Beholder<T> &rhs);

        /**
         * Sets the look at point of the beholder and increments the 
         * updateCounter. The Points position and lookAt must not be identical, 
         * and the vector (lookAt - position) must not be parallel to the up 
         * vector.
         *
         * Remark: The orthonormal coordinate system of the beholde is also 
         * updated which need multiple vector operations, including expensive 
         * vector normalisation.
         *
         * @param lookAt The new look at point for the beholder in world 
         *               coordinates
         *
         * @throws IllegalParamException if the distance between lookAt and 
         *         position is zero, or if the vector (lookAt - position) is
         *         parallel to the up vector.
         */
        template <class Sp> void SetLookAt(const math::AbstractPoint3D<T, Sp> &lookAt);

        /**
         * Sets the position of the beholder and increments the updateCounter.
         * The Points position and lookAt must not be identical and the vector
         * (lookAt - position) must not be parallel to the up vector.
         *
         * Remark: The orthonormal coordinate system of the beholde is also 
         * updated which need multiple vector operations, including expensive 
         * vector normalisation.
         *
         * @param position The new position for the beholder in world 
         *                 coordinates.
         *
         * @throws IllegalParamException if the distance between position and 
         *         lookAt is zero, or if the vector (lookAt - position) is
         *         parallel to the up vector.
         */
        template <class Sp> void SetPosition(const math::AbstractPoint3D<T, Sp> &position);

        /**
         * Sets the up vector of the beholder and increments the updateCounter.
         * The up vector will be normalized. The vector (lookAt - position) 
         * must not be parallel to the up vector, and the up vector must not be
         * a null vector.
         *
         * Remark: The orthonormal coordinate system of the beholde is also 
         * updated which need multiple vector operations, including expensive 
         * vector normalisation.
         *
         * @param up The new up vector for the beholder
         *
         * @throws IllegalParamException if the vector (lookAt - position) is
         *         parallel to the up vector, or if the up vector is a null 
         *         vector.
         */
        template <class Sp> void SetUpVector(const math::AbstractVector3D<T, Sp> &up);

        /**
         * Sets the view parameters (position, lookAt, and up vector) of the
         * beholder and increments the updateCounter. The up vector will be
         * normalized. The Points position and lookAt must not be identical, 
         * the vector (lookAt - position) must not be parallel to the up 
         * vector, and the up vector must not be a null vector.
         *
         * Remark: The orthonormal coordinate system of the beholde is also 
         * updated which need multiple vector operations, including expensive 
         * vector normalisation.
         *
         * @param position The new position for the beholder in world 
         *                 coordinates.
         * @param lookAt The new look at point for the beholder in world 
         *               coordinates
         * @param up The new up vector for the beholder
         *
         * @throws IllegalParamException if the distance between position and 
         *         lookAt is zero, if the vector (lookAt - position) is 
         *         parallel to the up vector, or if the up vector is a null 
         *         vector.
         */
        template <class Sp1, class Sp2, class Sp3> 
        void SetView(const math::AbstractPoint3D<T, Sp1> &position,
            const math::AbstractPoint3D<T, Sp2> &lookAt,
            const math::AbstractVector3D<T, Sp3> &up);

        /**
         * returns the position of the beholder in world coordinates.
         *
         * @return The position
         */
        inline const math::Point3D<T> & GetPosition(void) const {
            return this->position;
        }

        /**
         * returns the look at point of the beholder in world coordinates.
         *
         * @return The look at point
         */
        inline const math::Point3D<T> & GetLookAt(void) const {
            return this->lookAt;
        }

        /**
         * returns the front vector of the beholder.
         *
         * @return The front vector
         */
        inline const math::Vector3D<T> & GetFrontVector(void) const {
            return this->front;
        }

        /**
         * returns the right vector of the beholder.
         *
         * @return The right vector
         */
        inline const math::Vector3D<T> & GetRightVector(void) const {
            return this->right;
        }

        /**
         * returns the up vector of the beholder.
         *
         * @return The up vector
         */
        inline const math::Vector3D<T> & GetUpVector(void) const {
            return this->up;
        }

        /**
         * returns the value of the updateCounter.
         *
         * @return The value of the updateCounter
         */
        inline const unsigned int GetUpdateCounterValue(void) const {
            return this->updateCounter;
        }

    private:

        /**
         * Calculates front and right, based on position, lookAt and up.
         * Also can change the value of up to ensure orthogonality.
         */
        void CalcOrthoNormalVectors(void);

        /** position of the beholder in world coordinates */
        math::Point3D<T> position;

        /** look at point of the beholder in world coordinates */
        math::Point3D<T> lookAt;

        /**
         * Front vector of the beholder.
         */
        math::Vector3D<T> front;

        /**
         * Right vector of the beholder.
         */
        math::Vector3D<T> right;

        /** 
         * up vector of the beholder.
         * The vector (lookAt - position) and this vector must not be parallel.
         */
        math::Vector3D<T> up;

        /** number indicating updates of members */
        unsigned int updateCounter;
    };


    /*
     * Beholder::Beholder
     */
    template<class T> Beholder<T>::Beholder() : position(), 
            lookAt(static_cast<T>(0), static_cast<T>(0), static_cast<T>(-1)), 
            front(), right(), // front and right are calculated based on position, lookAt and up
            up(static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)),
            updateCounter(0) {
        this->CalcOrthoNormalVectors();
    };

    
    /*
     * Beholder::Beholder
     */
    template<class T> 
    template<class Sp1, class Sp2, class Sp3>
    Beholder<T>::Beholder(const math::AbstractPoint3D<T, Sp1> &position,
            const math::AbstractPoint3D<T, Sp2> &lookAt, 
            const math::AbstractVector3D<T, Sp3> &up) {
        this->updateCounter = 1;
        this->position = position;
        if (position != lookAt) {
            this->lookAt = lookAt;
        } else {
            this->updateCounter = 0;
            this->lookAt = position + math::Vector3D<T>(static_cast<T>(0), static_cast<T>(0), static_cast<T>(-1));
        }
        if ((up.IsNull()) || (up.IsParallel(this->position - this->lookAt))) {
            this->updateCounter = 0;
            this->up.SetComponents(0, 1, 0);
            if (up.IsParallel(this->up)) {
                this->up.SetComponents(0, 0, 1);
            }
        } else {
            this->up = up;
        }
        this->CalcOrthoNormalVectors();
    }


    /*
     * Beholder::Beholder
     */
    template<class T> Beholder<T>::Beholder(const Beholder &rhs) {
        this->operator=(rhs);
    }


    /*
     * Beholder::operator=
     */
    template<class T> Beholder<T>& Beholder<T>::operator=(const Beholder &rhs) {
        this->position = rhs.position;
        this->lookAt = rhs.lookAt;
        this->up = rhs.up;
        this->front = rhs.front;
        this->right = rhs.right;
        this->updateCounter = 0;
        return *this;
    }


    /*
     * Beholder::SetLookAt
     */
    template<class T> 
    template<class Sp> 
    void Beholder<T>::SetLookAt(const math::AbstractPoint3D<T, Sp> &lookAt) {
        if ((this->position == lookAt) || (this->up.IsParallel(this->position - lookAt))) {
            throw IllegalParamException("lookAt", __FILE__, __LINE__);
        }
        this->lookAt = lookAt;
        this->updateCounter++;
        this->CalcOrthoNormalVectors();
    }


    /*
     * Beholder::SetPosition
     */
    template<class T> 
    template<class Sp>
    void Beholder<T>::SetPosition(const math::AbstractPoint3D<T, Sp> &position) {
        if ((position == this->lookAt) || (this->up.IsParallel(position - this->lookAt))) {
            throw IllegalParamException("position", __FILE__, __LINE__);
        }
        this->position = position;
        this->updateCounter++;
        this->CalcOrthoNormalVectors();
    }


    /*
     * Beholder::SetUpVector
     */
    template<class T> 
    template<class Sp>
    void Beholder<T>::SetUpVector(const math::AbstractVector3D<T, Sp> &up) {
        if ((up.IsNull()) || (up.IsParallel(this->position - this->lookAt))) {
            throw IllegalParamException("up", __FILE__, __LINE__);
        }
        this->up = up;
        this->updateCounter++;
        this->CalcOrthoNormalVectors();
    }


    /*
     * Beholder::SetView
     */
    template<class T> 
    template<class Sp1, class Sp2, class Sp3>
    void Beholder<T>::SetView(const math::AbstractPoint3D<T, Sp1> &position, 
            const math::AbstractPoint3D<T, Sp2> &lookAt, 
            const math::AbstractVector3D<T, Sp3> &up) {
        if (position == lookAt) {
            throw IllegalParamException("position, or lookAt", __FILE__, __LINE__);
        }
        if (up.IsNull()) {
            throw IllegalParamException("up", __FILE__, __LINE__);
        }
        if (up.IsParallel(position - lookAt)) {
            throw IllegalParamException("up, position, or lookAt", __FILE__, __LINE__);
        }
        this->position = position;
        this->lookAt = lookAt;
        this->up = up;
        this->updateCounter++;
        this->CalcOrthoNormalVectors();
    }


    /*
     * Beholder<T>::CalcOrthoNormalVectors
     */
    template<class T> void Beholder<T>::CalcOrthoNormalVectors(void) {
        this->front = this->lookAt - this->position;
        this->front.Normalise();
        this->right = this->front.Cross(this->up);
        this->right.Normalise();
        this->up = this->right.Cross(this->front);
        this->up.Normalise();
    }

} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_BEHOLDER_H_INCLUDED */
