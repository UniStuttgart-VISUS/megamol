/*
 *	PlaneDataCall.h
 *
 *	Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 *	All rights reserved
 */

#ifndef MMPROTEINCUDAPLUGIN_PLANEDATACALL_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_PLANEDATACALL_H_INCLUDED

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/math/Plane.h"
#include "vislib/math/ShallowPlane.h"

namespace megamol {
namespace protein_cuda {

	class PlaneDataCall : public core::Call {
	public:

		/** Index of the GetExtent function */
		static const unsigned int CallForGetExtent;

		/** Index of the GetData function */
		static const unsigned int CallForGetData;

		/**
		 *	Answer the name of the objects of this description.
		 *	
		 *	@return The name of the objects of this description.
		 */
		static const char *ClassName(void) {
		    return "PlaneDataCall";
		}

		/**
		 *	Gets a human readable description of the module.
		 *	
		 *	@return A human readable description of the module.
		 */
		static const char *Description(void) {
		    return "Call to transmit one or more planes between two modules";
		}

		/**
		 *	Answer the number of functions used for this call.
		 *	
		 *	@return The number of functions used for this call.
		 */
		static unsigned int FunctionCount(void) {
		    return 2;
		}

		/**
		 *	Answer the name of the function used for this call.
		 *	
		 *	@param idx The index of the function to return it's name.
		 *	@return The name of the requested function.
		 */
		static const char * FunctionName(unsigned int idx) {
		    switch(idx) {
				case 0:
				    return "getData";
				case 1:
				    return "getExtent";
		    }
			return nullptr;
		}

		/** Ctor */
		PlaneDataCall(void);

		/** Dtor */
		virtual ~PlaneDataCall(void);

		/**
		 *	Returns the current data hash.
		 *	
		 *	@return The current data hash.
		 */
		SIZE_T DataHash(void);

		/**
		 *	Returns the plane count.
		 *
		 *	@return The plane count.
		 */
		unsigned int GetPlaneCnt(void);

		/**
		 *	Returns the plane data.
		 *
		 *	@return The plane data.
		 */
		const vislib::math::Plane<float> * GetPlaneData(void);

		/**
		 *	Sets the current data hash.
		 *	
		 *	@param dataHash The current data hash.
		 */
		void SetDataHash(SIZE_T dataHash);

		/**
		 *	Sets the number of available planes.
		 *
		 *	@param planeCnt The number of planes.
		 */
		void SetPlaneCnt(unsigned int planeCnt);

		/**
		 *	Sets the plane data.
		 *
		 *	@param planeData The Array containing the planes
		 */
		void SetPlaneData(const vislib::math::Plane<float> * planeData);

	private:

		/** The number of stored planes */
		unsigned int planeCnt;

		/** Pointer to the plane data */
		const vislib::math::Plane<float> * planeData;

		/** The current data hash */
		SIZE_T dataHash;
	};

	/** Description class typedef */
	typedef core::factories::CallAutoDescription<PlaneDataCall> PlaneDataCallDescription;

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif // #ifndef MMPROTEINCUDAPLUGIN_PLANEDATACALL_H_INCLUDED