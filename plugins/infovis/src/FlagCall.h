/*
 * FlagCall.h
 *
 * Author: Guido Reina
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_FLAGCALL_H_INCLUDED
#define MEGAMOL_FLAGCALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/Array.h"
#include "infovis/infovis.h"
#include "FlagStorage.h"

namespace megamol {
namespace infovis {

	/**
	 * call for passing flag data (FlagStorage) between modules
	 */

	class INFOVIS_API FlagCall : public megamol::core::Call {
	public:
		/**
		 * Answer the name of the objects of this description.
		 *
		 * @return The name of the objects of this description.
		 */
		static const char *ClassName(void) {
			return "FlagCall";
		}

		/**
		 * Gets a human readable description of the module.
		 *
		 * @return A human readable description of the module.
		 */
		static const char *Description(void) {
			return "Call to get index-synced flag data";
		}

		/** Index of the 'GetData' function */
		static const unsigned int CallForGetFlags;

		static const unsigned int CallForSetFlags;

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
				return "getFlags";
			case 1:
				return "setFlags";
			}
			return "";
		}

		inline const FlagStorage::FlagVectorType& GetFlags(void) const {
			return *this->flags;
		}

		inline bool has_data() const {
			return static_cast<bool>(flags);
		}

		/** warning: this steals the pointer from the caller so he cannot fiddle with its contents afterwards! */
		inline void SetFlags(std::shared_ptr<FlagStorage::FlagVectorType>& f) {
			this->flags = f;
			f.reset();
		}

		FlagCall(void);
		virtual ~FlagCall(void);

	private:
		std::shared_ptr<const FlagStorage::FlagVectorType> flags;
		// TODO less yucky
		friend class FlagStorage;
	};

	/** Description class typedef */
	typedef megamol::core::factories::CallAutoDescription<FlagCall> FlagCallDescription;

} /* end namespace protein_calls */
} /* end namespace megamol */

#endif /* MEGAMOL_FLAGCALL_H_INCLUDED */
