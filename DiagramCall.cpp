#include "stdafx.h"
#include "DiagramCall.h"

namespace megamol {
namespace protein {

/*
 * MolecularDataCall::CallForGetData
 */
const unsigned int protein::DiagramCall::CallForGetData = 0;

/*
 * Diagram2DCall::Diagram2DCall
 */
DiagramCall::DiagramCall(void) : theData() {
    // intentionally empty
}


/*
 * Diagram2DCall::~Diagram2DCall
 */
DiagramCall::~DiagramCall(void) {

}

} /* end namespace protein */
} /* end namespace megamol */
