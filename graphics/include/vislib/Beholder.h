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
#include "vislib/Point3D.h"
#include "vislib/Vector3D.h"


namespace vislib {
namespace graphics {

    /**
     * class modelling a 3d scene beholder
     *
     * TODO: change setter from Vector3D to AbstractVector3D
     * TODO: change setter from Point3D to AbstractPoint3D
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
         */
        Beholder(const math::Point3D<T> &position, 
            const math::Point3D<T> &lookAt, const math::Vector3D<T> &up);

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
         * @param lookAt The new look at point for the beholder in world 
         *               coordinates
         *
         * @throws IllegalParamException if the distance between lookAt and 
         *         position is zero, or if the vector (lookAt - position) is
         *         parallel to the up vector.
         */
        void SetLookAt(const math::Point3D<T> &lookAt);

        /**
         * Sets the position of the beholder and increments the updateCounter.
         * The Points position and lookAt must not be identical and the vector
         * (lookAt - position) must not be parallel to the up vector.
         *
         * @param position The new position for the beholder in world 
         *                 coordinates.
         *
         * @throws IllegalParamException if the distance between position and 
         *         lookAt is zero, or if the vector (lookAt - position) is
         *         parallel to the up vector.
         */
        void SetPosition(const math::Point3D<T> &position);

        /**
         * Sets the up vector of the beholder and increments the updateCounter.
         * The up vector will be normalized. The vector (lookAt - position) 
         * must not be parallel to the up vector, and the up vector must not be
         * a null vector.
         *
         * @param up The new up vector for the beholder
         *
         * @throws IllegalParamException if the vector (lookAt - position) is
         *         parallel to the up vector, or if the up vector is a null 
         *         vector.
         */
        void SetUpVector(const math::Vector3D<T> &up);

        /**
         * Sets the view parameters (position, lookAt, and up vector) of the
         * beholder and increments the updateCounter. The up vector will be
         * normalized. The Points position and lookAt must not be identical, 
         * the vector (lookAt - position) must not be parallel to the up 
         * vector, and the up vector must not be a null vector.
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
        void SetView(const math::Point3D<T> &position, 
            const math::Point3D<T> &lookAt, const math::Vector3D<T> &up);

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

        /** position of the beholder in world coordinates */
        math::Point3D<T> position;

        /** look at point of the beholder in world coordinates */
        math::Point3D<T> lookAt;

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
            up(static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)),
            updateCounter(0) {
    };

    
    /*
     * Beholder::Beholder
     */
    template<class T> Beholder<T>::Beholder(const math::Point3D<T> &position,
            const math::Point3D<T> &lookAt, const math::Vector3D<T> &up) {
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
            this->up.Normalize();
        }
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
        this->updateCounter = 0;
        return *this;
    }


    /*
     * Beholder::SetLookAt
     */
    template<class T> void Beholder<T>::SetLookAt(const math::Point3D<T> &lookAt) {
        if ((this->position == lookAt) || (this->up.IsParallel(this->position - lookAt))) {
            throw IllegalParamException("lookAt", __FILE__, __LINE__);
        }
        this->lookAt = lookAt;
    }


    /*
     * Beholder::SetPosition
     */
    template<class T> void Beholder<T>::SetPosition(const math::Point3D<T> &position) {
        if ((position == this->lookAt) || (this->up.IsParallel(position - this->lookAt))) {
            throw IllegalParamException("position", __FILE__, __LINE__);
        }
        this->position = position;
    }


    /*
     * Beholder::SetUpVector
     */
    template<class T> void Beholder<T>::SetUpVector(const math::Vector3D<T> &up) {
        if ((up.IsNull()) || (up.IsParallel(this->position - this->lookAt))) {
            throw IllegalParamException("up", __FILE__, __LINE__);
        }
        this->up = up;
        this->up.Normalize();
    }


    /*
     * Beholder::SetView
     */
    template<class T> void Beholder<T>::SetView(const math::Point3D<T> &position, 
            const math::Point3D<T> &lookAt, const math::Vector3D<T> &up) {
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
        this->up.Normalize();
    }


} /* end namespace graphics */
} /* end namespace vislib */

#endif /* VISLIB_BEHOLDER_H_INCLUDED */
