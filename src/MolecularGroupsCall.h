/*
 * MolecularGroupsCall.h
 *
 * Copyright (C) 2016 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_MOLECULARGROUPSCALL_H_INCLUDED
#define MMPROTEINPLUGIN_MOLECULARGROUPSCALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include <vector>

namespace megamol {
namespace protein {

	class MolecularGroupsCall : public megamol::core::AbstractGetData3DCall {
	public:

		/** Index of the 'GetData' function */
		static const unsigned int CallForGetData;

		/** Index of the 'GetExtent' function */
		static const unsigned int CallForGetExtent;

		/**
		 *	Answer the name of the objects of this description.
		 *	
		 *	@return The name of the objects of this description.
		 */
		static const char *ClassName(void) {
			return "MolecularGroupsCall";
		}

		/**
		 * Answer a human readable description of this module.
		 *
		 * @return A human readable description of this module.
		 */
		static const char *Description(void) {
			return "Call for sending the information which molecules of a MolecularDataCall are equal";
		}

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
        static const char * FunctionName(unsigned int idx) {
            switch(idx) {
                case 0:
                    return "GetData";
                case 1:
                    return "GetExtend";
            }
			return "";
        }

		/** Ctor. */
		MolecularGroupsCall(void) {
		}

		/** Dtor. */
		virtual ~MolecularGroupsCall(void) {
		}

		// -------------------- get and set routines --------------------

		/**
		 *	Set the number of groups.
		 *
		 *	@param cnt The number of groups.
		 */
		void SetGroupCount(unsigned int cnt) { this->groupCount = cnt; }

		/**
		 *	Get the number of groups.
		 *
		 *	@return The number of groups.
		 */
		const unsigned int GroupCount() const { return this->groupCount; }

		/**
		 *	Assign the group sizes.
		 *
		 *	@param sizes The size list.
		 */
		void SetGroupSizes(unsigned int * sizes) { this->groupSizes = sizes; }

		/**
		 *	Access the group size list.
		 *
		 *	@return The group size list.
		 */
		const unsigned int * GroupSizes() const { return this->groupSizes; }

		/**
		 *	Assign the group data list.
		 *
		 *	@param data The group data.
		 */
		void SetGroupData(int** data) { this->groupData = data; }

		/**
		 *	Access the group data list.
		 *
		 *	@return The group data list.
		 */
		int** Groups() { return this->groupData; }

		/**
		 *	Sets the calltime to request data for.
		 *
		 *	@param calltime The calltime to request data for.
		 */
		void SetCalltime(float calltime) {
			this->calltime = calltime;
		}

		/**
		 *	Answer the calltime.
		 *
		 *	@return The calltime.
		 */
		float Calltime(void) const {
			return this->calltime;
		}

	private:
		// -------------------- variables --------------------

		/** Stores the number of groups */
		unsigned int groupCount;

		/** List containing the group size for each group */
		unsigned int * groupSizes;

		/** List of groups */
		int** groupData;

		/** The exact requested/stored calltime */
		float calltime;
	};

	/** Description class typedef */
	typedef megamol::core::factories::CallAutoDescription<MolecularGroupsCall> MolecularGroupsCallDescription;

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_MOLECULARGROUPSCALL_H_INCLUDED */