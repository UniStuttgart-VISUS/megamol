#include "CPUQuery.h"

namespace megamol::frontend_resources::performance {
CPUQuery::CPUQuery() {}

CPUQuery::~CPUQuery() {}

void CPUQuery::Counter() {
    this->value_ = time_point::clock::now();
}

time_point CPUQuery::GetNW() {
    return value_;
}
} // namespace megamol::frontend_resources::performance
