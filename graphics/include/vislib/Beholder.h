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
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/graphicstypes.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Point.h"
#include "vislib/Vector.h"


namespace vislib {
namespace graphics {

    /**
     * class modelling a 3d scene beholder
     */
    class Beholder {
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
        template <class Tp1, class Sp1, class Tp2, class Sp2, class Tp3, class Sp3> 
        Beholder(const math::AbstractPoint<Tp1, 3, Sp1> &position, 
            const math::AbstractPoint<Tp2, 3, Sp2> &lookAt, 
            const math::AbstractVector<Tp3, 3, Sp3> &up);

        /**
         * copy ctor
         *
         * copies all values from rhs but updateCounter which is set to 0.
         *
         * @param rhs The object to be cloned.
         */
        Beholder(const Beholder &rhs);

        /**
         * assignment operator
         *
         * copies all values from rhs but updateCounter which is set to 0.
         *
         * @param rhs The object to be cloned.
         *
         * @return reference to this
         */
        Beholder& operator=(const Beholder &rhs);

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
        template <class Tp, class Sp> 
        void SetLookAt(const math::AbstractPoint<Tp, 3, Sp> &lookAt);

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
        template <class Tp, class Sp> 
        void SetPosition(const math::AbstractPoint<Tp, 3, Sp> &position);

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
        template <class Tp, class Sp> 
        void SetUpVector(const math::AbstractVector<Tp, 3, Sp> &up);

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
        template <class Tp1, class Sp1, class Tp2, class Sp2, class Tp3, class Sp3> 
        void SetView(const math::AbstractPoint<Tp1, 3, Sp1> &position,
            const math::AbstractPoint<Tp2, 3, Sp2> &lookAt,
            const math::AbstractVector<Tp3, 3, Sp3> &up);

        /**
         * returns the position of the beholder in world coordinates.
         *
         * @return The position
         */
        inline const math::Point<SceneSpaceType, 3> & GetPosition(void) const {
            return this->position;
        }

        /**
         * returns the look at point of the beholder in world coordinates.
         *
         * @return The look at point
         */
        inline const math::Point<SceneSpaceType, 3> & GetLookAt(void) const {
            return this->lookAt;
        }

        /**
         * returns the normalized front vector of the beholder.
         *
         * @return The front vector
         */
        inline const math::Vector<SceneSpaceType, 3> & GetFrontVector(void) const {
            return this->front;
        }

        /**
         * returns the normalized right vector of the beholder.
         *
         * @return The right vector
         */
        inline const math::Vector<SceneSpaceType, 3> & GetRightVector(void) const {
            return this->right;
        }

        /**
         * returns the normalized up vector of the beholder.
         *
         * @return The up vector
         */
        inline const math::Vector<SceneSpaceType, 3> & GetUpVector(void) const {
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
        math::Point<SceneSpaceType, 3> position;

        /** look at point of the beholder in world coordinates */
        math::Point<SceneSpaceType, 3> lookAt;

        /** Front vector of the beholder. */
        math::Vector<SceneSpaceType, 3> front;

        /** Right vector of the beholder. */
        math::Vector<SceneSpaceType, 3> right;

        /** 
         * up vector of the beholder.
         * The vector (lookAt - position) and this vector must not be parallel.
         */
        math::Vector<SceneSpaceType, 3> up;

        /** number indicating updates of members */
        unsigned int updateCounter;
    };

    
    /*
     * Beholder::Beholder
     */
    template<class Tp1, class Sp1, class Tp2, class Sp2, class Tp3, class Sp3>
    Beholder::Beholder(const math::AbstractPoint<Tp1, 3, Sp1> &position,
            const math::AbstractPoint<Tp2, 3, Sp2> &lookAt, 
            const math::AbstractVector<Tp3, 3, Sp3> &up) {
        this->updateCounter = 1;
        this->position = position;
        if (position != lookAt) {
            this->lookAt = lookAt;
        } else {
            this->updateCounter = 0;
            this->lookAt = position 
                + math::Vector<SceneSpaceType, 3>(0, 0, -1);
        }
        if ((up.IsNull()) || (up.IsParallel(this->position - this->lookAt))) {
            this->updateCounter = 0;
            this->up.Set(0, 1, 0);
            if (up.IsParallel(this->up)) {
                this->up.Set(0, 0, 1);
            }
        } else {
            this->up = up;
        }
        this->CalcOrthoNormalVectors();
    }




    /*
     * Beholder::SetLookAt
     */
    template<class Tp, class Sp> 
    void Beholder::SetLookAt(const math::AbstractPoint<Tp, 3, Sp> &lookAt) {
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
    template<class Tp, class Sp>
    void Beholder::SetPosition(const math::AbstractPoint<Tp, 3, Sp> &position) {
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
    template<class Tp, class Sp>
    void Beholder::SetUpVector(const math::AbstractVector<Tp, 3, Sp> &up) {
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
    template<class Tp1, class Sp1, class Tp2, class Sp2, class Tp3, class Sp3>
    void Beholder::SetView(const math::AbstractPoint<Tp1, 3, Sp1> &position, 
            const math::AbstractPoint<Tp2, 3, Sp2> &lookAt, 
            const math::AbstractVector<Tp3, 3, Sp3> &up) {
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

} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_BEHOLDER_H_INCLUDED */
