/*
 * Vector3fParam.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VECTOR3FPARAM_H_INCLUDED
#define MEGAMOLCORE_VECTOR3FPARAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"
#include "AbstractParam.h"
#include "vislib/math/Vector.h"
#include <array>


namespace megamol {
namespace core {
namespace param {

    /**
     * class for (float)vector parameter objects
     * (delimiter for the float values is ';')
     */
    class MEGAMOLCORE_API Vector3fParam : public AbstractParam {
    public:

        /**
         * Ctor.
         *
         * @param initVal The initial value
         */
        Vector3fParam(const vislib::math::Vector<float, 3> &initVal);

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param minVal The minimum value
         */
        Vector3fParam(const vislib::math::Vector<float, 3> &initVal,
            const vislib::math::Vector<float, 3> &minVal);

        /**
         * Ctor.
         *
         * @param initVal The initial value
         * @param minVal The minimum value
         * @param maxVal The maximum value
         */
        Vector3fParam(const vislib::math::Vector<float, 3> &initVal,
            const vislib::math::Vector<float, 3> &minVal,
            const vislib::math::Vector<float, 3> &maxVal);

        /**
         * Dtor.
         */
        virtual ~Vector3fParam(void);

        std::array<float, 3> getArray() const { 
            return std::array<float, 3>({this->Value().GetX(), this->Value().GetY(), this->Value().GetZ()});
        }

        /**
         * Returns a machine-readable definition of the parameter.
         *
         * @param outDef A memory block to receive a machine-readable
         *               definition of the parameter.
         */
        virtual void Definition(vislib::RawStorage& outDef) const;

        /**
         * Tries to parse the given string as value for this parameter and
         * sets the new value if successful. This also triggers the update
         * mechanism of the slot this parameter is assigned to.
         *
         * @param v The new value for the parameter as string.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool ParseValue(const vislib::TString& v);

        /**
         * Sets the value of the parameter and optionally sets the dirty flag
         * of the owning parameter slot.
         *
         * @param v the new value for the parameter
         * @param setDirty If 'true' the dirty flag of the owning parameter
         *                 slot is set and the update callback might be called.
         */
        void SetValue(const vislib::math::Vector<float, 3>& v, bool setDirty = true);

        /**
         * Gets the value of the parameter
         *
         * @return The value of the parameter
         */
        inline const vislib::math::Vector<float,3>& Value(void) const {
            return this->val;
        }
        /**
         * Needed for RemoteControl - Manuel Gräber
         * Gets the minimum value of the parameter
         *
         * @return The minimum value of the parameter
         */
        inline const vislib::math::Vector<float,3>& MinValue(void) const {
            return this->minVal;
        }

        /**
         * Needed for RemoteControl - Manuel Gräber
         * Gets the maximum value of the parameter
         *
         * @return The maximum value of the parameter
         */
        inline const vislib::math::Vector<float,3>& MaxValue(void) const {
            return this->maxVal;
        }
        /**
         * Returns the value of the parameter as string.
         *
         * @return The value of the parameter as string.
         */
        virtual vislib::TString ValueString(void) const;

        /**
         * Gets the value of the parameter
         *
         * @return The value of the parameter
         */
        inline operator const vislib::math::Vector<float,3>&(void) const {
            return this->val;
        }

    private:

        /**
         * 'True' if vector A is less or equal than vector B.
         *
         * @param A The left hand side operand
         * @param B The right hand side operand
         *
         * @ return 'True' if A <= B, 'false' otherwise.
         */
        bool isLessOrEqual(const vislib::math::Vector<float, 3> &A,
            const vislib::math::Vector<float, 3> &B) const;

        /**
         * 'True' if vector A is greater or equal than vector B.
         *
         * @param A The left hand side operand
         * @param B The right hand side operand
         *
         * @ return 'True' if A >= B, 'false' otherwise.
         */
        bool isGreaterOrEqual(const vislib::math::Vector<float, 3> &A,
            const vislib::math::Vector<float, 3> &B) const;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** The value of the parameter */
        vislib::math::Vector<float, 3> val;

        /** The minimum value for the parameter */
        vislib::math::Vector<float, 3> minVal;

        /** The maximum value for the parameter */
        vislib::math::Vector<float, 3> maxVal;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

    };

} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VECTOR3FPARAM_H_INCLUDED */
