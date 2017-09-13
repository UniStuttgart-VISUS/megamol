#include "stdafx.h"
#include "SplitMergeFeature.h"

namespace megamol {
namespace protein_cuda {

SplitMergeFeature::SplitMergeFeature(float maxT, vislib::math::Vector<float, 3> pos) : protein_calls::SplitMergeCall::SplitMergeMappable(),
        data(), maxTime( maxT), maxSurfaceArea( 0.0f), position( pos) {
    data.AssertCapacity( 1000);
    data.SetCapacityIncrement( 100);
}


SplitMergeFeature::~SplitMergeFeature(void) {
}

int SplitMergeFeature::GetDataCount() const {
    return static_cast<int>(data.Count());
}

bool SplitMergeFeature::GetAbscissaValue(const SIZE_T index, float *value) const {
    if (data[index] != NULL) {
        *value = data[index]->First();
        return true;
    } else {
        return false;
    }
}

float SplitMergeFeature::GetOrdinateValue(const SIZE_T index) const {
    if (data[index] != NULL) {
        return data[index]->Second();
    }
    return 0.0f;
}

vislib::Pair<float, float> SplitMergeFeature::GetAbscissaRange() const {
    return vislib::Pair<float, float>(0.0f, this->maxTime);
}

vislib::Pair<float, float> SplitMergeFeature::GetOrdinateRange() const {
    return vislib::Pair<float, float>(0.0f, this->maxSurfaceArea);
}

void SplitMergeFeature::AppendValue( vislib::Pair<float, float> p) { 
    this->data.Append(new vislib::Pair<float, float>(p));
    if( p.Second() > this->maxSurfaceArea ) 
        this->maxSurfaceArea = p.Second();
}


} /* namespace protein_cuda */
} /* namespace megamol */
