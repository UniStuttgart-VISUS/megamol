#include "stdafx.h"
#include "mmcore/moldyn/DiagramCall.h"

namespace megamol {
namespace core {
namespace moldyn {

	/*
	 * MolecularDataCall::CallForGetData
	 */
	const unsigned int core::moldyn::DiagramCall::CallForGetData = 0;

	/*
	 * Diagram2DCall::Diagram2DCall
	 */
	DiagramCall::DiagramCall(void) {
		this->theData = new vislib::Array<DiagramSeries*>();
		this->guides = new vislib::PtrArray<DiagramGuide>();
	}


	/*
	 * Diagram2DCall::~Diagram2DCall
	 */
	DiagramCall::~DiagramCall(void) {
		delete this->theData;
		delete this->guides;
	}

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */
