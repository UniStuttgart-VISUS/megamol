/*
 * VolumeMeshRenderer.h
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MAPPABLEFLOATPAIR_H_INCLUDED
#define MEGAMOLCORE_MAPPABLEFLOATPAIR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/Pair.h"
#include "protein_calls/DiagramCall.h"
#include "protein_calls/SplitMergeCall.h"

namespace megamol {
namespace protein_cuda {

class MappableFloatPair : public protein_calls::DiagramCall::DiagramMappable {
public:
    MappableFloatPair(float offsetX = 0.0f, float offsetY = 0.0f, bool flipX = false, int holePos = -1);
    ~MappableFloatPair(void);

    virtual int GetAbscissaeCount() const;
    virtual int GetDataCount() const;
    virtual bool IsCategoricalAbscissa(const SIZE_T abscissa) const;
    virtual bool GetAbscissaValue(const SIZE_T index, const SIZE_T abscissaIndex, vislib::StringA *category) const;
    virtual bool GetAbscissaValue(const SIZE_T index, const SIZE_T abscissaIndex, float *value) const;
    virtual float GetOrdinateValue(const SIZE_T index) const;
    virtual vislib::Pair<float, float> GetAbscissaRange(const SIZE_T abscissaIndex) const;
    virtual vislib::Pair<float, float> GetOrdinateRange() const;

private:
    float offsetX;
    float offsetY;
    bool flipX;
    int holePos;
};

class MappableWibble : public protein_calls::SplitMergeCall::SplitMergeMappable {
public:

    MappableWibble(int holePos = -1);
    ~MappableWibble(void);

    virtual int GetDataCount() const;
    virtual bool GetAbscissaValue(const SIZE_T index, float *value) const;
    virtual float GetOrdinateValue(const SIZE_T index) const;
    virtual vislib::Pair<float, float> GetAbscissaRange() const;
    virtual vislib::Pair<float, float> GetOrdinateRange() const;

private:
    int holePos;
};

} /* namespace protein_cuda */
} /* namespace megamol */

#endif /* MEGAMOLCORE_MAPPABLEFLOATPAIR_H_INCLUDED */
