#include "stdafx.h"
#include "DiagramCall.h"

namespace megamol {
namespace protein_cuda {

/*
 * MolecularDataCall::CallForGetData
 */
const unsigned int protein_cuda::DiagramCall::CallForGetData = 0;

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

} /* end namespace protein_cuda */
} /* end namespace megamol */
