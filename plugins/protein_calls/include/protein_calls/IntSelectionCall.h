/*
 * SplitMergeCall.h
 *
 * Author: Guido Reina
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_PROTEIN_CALL_SELECTIONCALL_H_INCLUDED
#define MEGAMOL_PROTEIN_CALL_SELECTIONCALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/Array.h"

namespace megamol {
namespace protein_calls {

	/**
	 * Base class for graph calls and data interfaces.
	 *
	 * Graphs based on coordinates can contain holes where the respective
	 * getters return false for the abscissae. For categorical graphs this
	 * seems useless as the abscissa is sparse anyway, but the interface
	 * allows for that as well.
	 */

	class IntSelectionCall : public megamol::core::Call {
	public:
		/**
		 * Answer the name of the objects of this description.
		 *
		 * @return The name of the objects of this description.
		 */
		static const char *ClassName(void) {
			return "IntSelectionCall";
		}

		/**
		 * Gets a human readable description of the module.
		 *
		 * @return A human readable description of the module.
		 */
		static const char *Description(void) {
			return "Call to get selection IDs";
		}

		/** Index of the 'GetData' function */
		static const unsigned int CallForGetSelection;

		static const unsigned int CallForSetSelection;

		/**
		 * Answer the number of functions used for this call.
		 *
		 * @return The number of functions used for this call.
		 */
		static unsigned int FunctionCount(void) {
			return 2;
		}

		/**
		 * Answer the name of the function used for this call.
		 *
		 * @param idx The index of the function to return it's name.
		 *
		 * @return The name of the requested function.
		 */
		static const char* FunctionName(unsigned int idx) {
			switch (idx) {
			case 0:
				return "getSelection";
			case 1:
				return "setSelection";
			}
			return "";
		}

		inline vislib::Array<int> *GetSelectionPointer(void) const {
			return this->selection;
		}

		inline void SetSelectionPointer(vislib::Array<int> *selection) {
			this->selection = selection;
		}

		IntSelectionCall(void);
		virtual ~IntSelectionCall(void);

	private:
		vislib::Array<int> *selection;
	};

	/** Description class typedef */
	typedef megamol::core::factories::CallAutoDescription<IntSelectionCall> IntSelectionCallDescription;

} /* end namespace protein_calls */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTEIN_CALL_SELECTIONCALL_H_INCLUDED */
