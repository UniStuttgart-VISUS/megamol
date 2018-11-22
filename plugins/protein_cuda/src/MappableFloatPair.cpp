#include "stdafx.h"
#include "MappableFloatPair.h"

namespace megamol {
namespace protein_cuda {

MappableFloatPair::MappableFloatPair(float offsetX, float offsetY, bool flipX, int holePos) : protein_calls::DiagramCall::DiagramMappable(),
    offsetX(offsetX), offsetY(offsetY), flipX(flipX), holePos(holePos) {
}


MappableFloatPair::~MappableFloatPair(void) {
}

int MappableFloatPair::GetAbscissaeCount() const {
    return 1;
}

int MappableFloatPair::GetDataCount() const {
    return 10;
}

bool MappableFloatPair::IsCategoricalAbscissa(const SIZE_T abscissa) const {
    return false;
}

bool MappableFloatPair::GetAbscissaValue(const SIZE_T index, const SIZE_T abscissaIndex, vislib::StringA *category) const {
    *category = vislib::StringA("cat");
    if (index == holePos) {
        return false;
    } else {
        return true;
    }
}

bool MappableFloatPair::GetAbscissaValue(const SIZE_T index, const SIZE_T abscissaIndex, float *value) const {
    *value = ((index / static_cast<float>(this->GetDataCount())))
        * (this->GetAbscissaRange(0).Second() - this->GetAbscissaRange(0).First()) + this->GetAbscissaRange(0).First();
    if (index == holePos) {
        return false;
    } else {
        return true;
    }
}

float MappableFloatPair::GetOrdinateValue(const SIZE_T index) const {
    float x = ((flipX ? 1.0f : 0.0f) +
        (flipX ? -1.0f : 1.0f) * (index / static_cast<float>(this->GetDataCount())) + offsetX);
    x *= static_cast<float>(this->GetAbscissaRange(0).Second() - this->GetAbscissaRange(0).First()) + this->GetAbscissaRange(0).First();
    return vislib::math::Sqrt(x) + 0.2f*sin(x*10) + offsetY;
}

vislib::Pair<float, float> MappableFloatPair::GetAbscissaRange(const SIZE_T abscissaIndex) const {
    return vislib::Pair<float, float>(1.0f, 10.0f);
}

vislib::Pair<float, float> MappableFloatPair::GetOrdinateRange() const {
    return vislib::Pair<float, float>(0.0f, 5.0f);;
}


MappableWibble::MappableWibble(int holePos) : holePos(holePos) {

}


MappableWibble::~MappableWibble(void) {

}


int MappableWibble::GetDataCount() const {
    return 10;
}


bool MappableWibble::GetAbscissaValue(const SIZE_T index, float *value) const {
	*value = static_cast<float>(index);
    if (index == holePos) {
        return false;
    } else {
        return true;
    }
}


float MappableWibble::GetOrdinateValue(const SIZE_T index) const {
    return sin(static_cast<float>(index)) + 1.0f;
}


vislib::Pair<float, float> MappableWibble::GetAbscissaRange() const {
    return vislib::Pair<float, float>(0.0f, 9.0f);
}


vislib::Pair<float, float> MappableWibble::GetOrdinateRange() const {
    return vislib::Pair<float, float>(0.0f, 2.0f);
}


} /* namespace protein_cuda */
} /* namespace megamol */
