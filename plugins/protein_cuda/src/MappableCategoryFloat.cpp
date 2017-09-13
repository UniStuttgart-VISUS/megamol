#include "stdafx.h"
#include "MappableCategoryFloat.h"

namespace megamol {
namespace protein_cuda {

MappableCategoryFloat::MappableCategoryFloat(int instance) : instance(instance) {
}


MappableCategoryFloat::~MappableCategoryFloat(void) {
}

int MappableCategoryFloat::GetAbscissaeCount() const {
    return 1;
}

int MappableCategoryFloat::GetDataCount() const {
    return 3;
}

bool MappableCategoryFloat::IsCategoricalAbscissa(const SIZE_T abscissa) const {
    return true;
}

bool MappableCategoryFloat::GetAbscissaValue(const SIZE_T index, const SIZE_T abscissaIndex, vislib::StringA *category) const {
    switch(index) {
    case 0:
        switch(instance) {
            case 0:
                *category = vislib::StringA("hurz");
                break;
            case 1:
                *category = vislib::StringA("heinz");
                break;
        }
        break;
    case 1:
        *category = vislib::StringA("hugo");
        break;
    case 2:
        switch(instance) {
            case 0:
                *category = vislib::StringA("heinz");
                break;
            case 1:
                *category = vislib::StringA("horscht");
                break;
        }
        break;
    default:
        *category = vislib::StringA::EMPTY;
        break;
    }
    return true;
}

bool MappableCategoryFloat::GetAbscissaValue(const SIZE_T index, const SIZE_T abscissaIndex, float *value) const {
    *value = 0.0f;
    return true;
}

float MappableCategoryFloat::GetOrdinateValue(const SIZE_T index) const {
    float ret;
    switch(index) {
    case 0:
        ret = 1.0f;
        break;
    case 1:
        ret = 2.0f;
        break;
    case 2:
        ret = 3.0f;
        break;
    default:
        ret = 0.0f;
        break;
    }
    return ret;
}

vislib::Pair<float, float> MappableCategoryFloat::GetAbscissaRange(const SIZE_T abscissaIndex) const {
    return vislib::Pair<float, float>(0.0f, 0.0f);
}

vislib::Pair<float, float> MappableCategoryFloat::GetOrdinateRange() const {
    return vislib::Pair<float, float>(-1.0f, 10.0f);;
}

} /* namespace protein_cuda */
} /* namespace megamol */
