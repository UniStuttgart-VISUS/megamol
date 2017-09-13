/*
 * SplitMergeFeature.h
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLPROTEIN_SPLITMERGEFEATURE_H_INCLUDED
#define MEGAMOLPROTEIN_SPLITMERGEFEATURE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "protein_calls/SplitMergeCall.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/math/Vector.h"

namespace megamol {
namespace protein_cuda {

    /**
     * Split-Merge Feature
     */
	class SplitMergeFeature : public protein_calls::SplitMergeCall::SplitMergeMappable
    {
    public:
        SplitMergeFeature( float maxT, vislib::math::Vector<float, 3> pos = vislib::math::Vector<float, 3>(0, 0, 0));
        ~SplitMergeFeature(void);

        virtual int GetDataCount() const;
        virtual bool GetAbscissaValue(const SIZE_T index, float *value) const;
        virtual float GetOrdinateValue(const SIZE_T index) const;
        virtual vislib::Pair<float, float> GetAbscissaRange() const;
        virtual vislib::Pair<float, float> GetOrdinateRange() const;
        
        /** 
         * Append a value pair (abscissa and ordinate value) to the list of values.
         *
         * @param x The abscissa value.
         * @param y The ordinate value.
         */
        VISLIB_FORCEINLINE void AppendValue( float x, float y) {
            this->AppendValue( vislib::Pair<float, float>( x, y));
        }
        
        /** 
         * Append a value pair (abscissa and ordinate value) to the list of values.
         *
         * @param p The value pair.
         */
        void AppendValue( vislib::Pair<float, float> p);
        
        /** 
         * Append a NULL value pair signaling a hole.
         */
        void AppendHole() {
            this->data.Append( NULL);
        };
        
        /** 
         * Set the position of the feature (e.g. centroid).
         *
         * @param pos The position.
         */
        VISLIB_FORCEINLINE void SetPosition( vislib::math::Vector<float, 3> pos) {
            this->position = pos;
        }
        
        /** 
         * Set the position of the feature (e.g. centroid).
         *
         * @param x The x component of the position.
         * @param y The y component of the position.
         * @param z The z component of the position.
         */
        VISLIB_FORCEINLINE void SetPosition( const float x, const float y, const float z) {
            this->position.Set( x, y, z);
        }
        
        /** 
         * Get the position of the feature.
         *
         * @return The position.
         */
        VISLIB_FORCEINLINE vislib::math::Vector<float, 3> GetPosition() {
            return this->position;
        }

    private:
        vislib::PtrArray<vislib::Pair<float, float> > data;
        float maxTime;
        float maxSurfaceArea;
        vislib::math::Vector<float, 3> position;
    };


} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif // MEGAMOLPROTEIN_SPLITMERGEFEATURE_H_INCLUDED
