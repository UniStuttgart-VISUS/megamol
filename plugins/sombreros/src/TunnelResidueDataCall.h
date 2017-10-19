/*
 * TunnelResidueDataCall.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMSOMBREROSPLUGIN_TUNNELRESIDUEDATACALL_H_INCLUDED
#define MMSOMBREROSPLUGIN_TUNNELRESIDUEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractGetData3DCall.h"

namespace megamol {
namespace sombreros {

	class TunnelResidueDataCall : public core::AbstractGetData3DCall {
	public:

		/** Struct representing a single tunnel */
		struct Tunnel
		{
		public:
			/** Ctor. */
			Tunnel(void) {
				numCoordinates = 0;
				coordinates = nullptr;
				atomNumbers = nullptr;
				firstAtomIndices = nullptr;
				atomIdentifiers = nullptr;
			}
			/** array storing the voronoi vertex locations of the tunnel represented */
			float * coordinates;

			/** the number of voronoi vertex coordinates of the tunnel */
			int numCoordinates;

			/** the numbers of stored atoms per voronoi vertex. length: numCoordinates */
			int * atomNumbers;

			/** the indices of the first atoms belonging to each voronoi vertex. length: numCoordinates */
			int * firstAtomIndices;

			/** the identifiers for all atoms stored in the call. first: identifier, second: number of occurences */
			std::pair<int, int> * atomIdentifiers;
		};

		/**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "TunnelResidueDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call transporting tunnel data alongside with redidue indices beside it";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return AbstractGetData3DCall::FunctionCount();
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            return AbstractGetData3DCall::FunctionName(idx);
        }

		/** Ctor. */
		TunnelResidueDataCall(void);

		/** Dtor. */
		virtual ~TunnelResidueDataCall(void);

		/**
		 * Returns the number of tunnels stored in this call
		 *
		 * @return The number of tunnels.
		 */
		int getTunnelNumber(void) const {
			return this->numTunnels;
		}

		/**
		 * Sets the number of tunnels
		 *
		 * @param numTunnels The new number of tunnels
		 */
		void setTunnelNumber(const int numTunnels) {
			this->numTunnels = numTunnels;
		}

		/**
		 * Returns the tunnel descriptions
		 *
		 * @param return The tunnel descriptions
		 */
		const Tunnel * getTunnelDescriptions(void) const {
			return this->tunnels;
		}

		/**
		 * Sets the new tunnel description pointer
		 *
		 * @param tunnels The pointer to the tunnel descriptions
		 */
		void setTunnelDescriptions(Tunnel * tunnels) {
			this->tunnels = tunnels;
		}

	private:

		/** array containing all tunnels */
		Tunnel * tunnels;

		/** the number of tunnels stored in this call */
		int numTunnels;
	};

	/** Description class typedef */
	typedef megamol::core::factories::CallAutoDescription<TunnelResidueDataCall> TunnelResidueDataCallDescription;

} /* end namespace sombreros */
} /* end namespace megamol */

#endif