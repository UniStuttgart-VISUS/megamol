#include "stdafx.h"
#include "MolecularSurfaceFeature.h"

namespace megamol {
namespace protein_cuda {

MolecularSurfaceFeature::MolecularSurfaceFeature( float maxT, vislib::math::Vector<float, 3> pos) : protein_calls::DiagramCall::DiagramMappable(),
        data(), maxTime( maxT), maxSurfaceArea( 0.0f), position( pos) {
    data.SetCapacityIncrement( 1000);
}


MolecularSurfaceFeature::~MolecularSurfaceFeature(void) {
}

int MolecularSurfaceFeature::GetAbscissaeCount() const {
    return 1;
}

int MolecularSurfaceFeature::GetDataCount() const {
    return static_cast<int>(data.Count());
}

bool MolecularSurfaceFeature::IsCategoricalAbscissa(const SIZE_T abscissa) const {
    return false;
}

bool MolecularSurfaceFeature::GetAbscissaValue(const SIZE_T index, const SIZE_T abscissaIndex, vislib::StringA *category) const {
    if (data[index] != NULL) {
        *category = vislib::StringA("cat");
        return true;
    } else {
        return false;
    }
}

bool MolecularSurfaceFeature::GetAbscissaValue(const SIZE_T index, const SIZE_T abscissaIndex, float *value) const {
    if (data[index] != NULL) {
        *value = data[index]->First();
        return true;
    } else {
        return false;
    }
}

float MolecularSurfaceFeature::GetOrdinateValue(const SIZE_T index) const {
    if (data[index] != NULL) {
        return data[index]->Second();
    }
    return 0.0f;
}

vislib::Pair<float, float> MolecularSurfaceFeature::GetAbscissaRange(const SIZE_T abscissaIndex) const {
    return vislib::Pair<float, float>(0.0f, this->maxTime);
}

vislib::Pair<float, float> MolecularSurfaceFeature::GetOrdinateRange() const {
    return vislib::Pair<float, float>(0.0f, this->maxSurfaceArea);
}

void MolecularSurfaceFeature::AppendValue( vislib::Pair<float, float> p) { 
    this->data.Append(new vislib::Pair<float, float>(p));
    if( p.Second() > this->maxSurfaceArea ) 
        this->maxSurfaceArea = p.Second();
}


} /* namespace protein_cuda */
} /* namespace megamol */
