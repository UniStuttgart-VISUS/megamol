/*
 * STLWriter.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_IO_STLWRITER_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_STLWRITER_H_INCLUDED
#pragma once

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/DataWriterCtrlCall.h"

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/param/ParamSlot.h"

#include "geometry_calls/CallTriMeshData.h"

#include <string>
#include <tuple>

namespace megamol
{
	namespace stdplugin
	{
		namespace datatools
		{
			namespace io
			{
				class STLWriter : public core::AbstractDataWriter
				{
				public:
					/// <summary>
					/// Answer the name of this module
					/// </summary>
					/// <returns>Module name</returns>
					static const char* ClassName()
					{
						return "STLWriter";
					}

					/// <summary>
					/// Answer a human-readable description
					/// </summary>
					/// <returns>Description</returns>
					static const char* Description()
					{
						return "Writer for triangle data in STL files";
					}

					/// <summary>
					/// Answer the availability of this module
					/// </summary>
					/// <returns>Availability</returns>
					static bool IsAvailable()
					{
						return true;
					}

					/// <summary>
					/// Disallow quickstart (I don't know why, though)
					/// </summary>
					/// <returns>Quickstart supported?</returns>
					static bool SupportQuickstart()
					{
						return false;
					}

					/// <summary>
					/// Constructor
					/// </summary>
					STLWriter();

					/// <summary>
					/// Destructor
					/// </summary>
					~STLWriter();

				protected:

					/// <summary>
					/// Create the module
					/// </summary>
					/// <returns>True on success; false otherwise</returns>
					virtual bool create() override;

					/// <summary>
					/// Callback function for requesting capability information
					/// </summary>
					/// <param name="caller">Call for this request</param>
					/// <returns>True on success; false otherwise</returns>
					virtual bool getCapabilities(core::DataWriterCtrlCall& call) override;

					/// <summary>
					/// Callback function for storing data to file
					/// </summary>
					/// <returns>True on success; false otherwise</returns>
					virtual bool run() override;

					/// <summary>
					/// Release the module
					/// </summary>
					virtual void release() override;

				private:
					/// <summary>
					/// Class for wrapping a pointer, using a pointer or a container as input
					/// </summary>
					/// <typeparam name="VT">Value type</typeparam>
					/// <typeparam name="CT">Container type</typeparam>
					template <typename VT, typename CT>
					class pointer_wrapper
					{
					public:
						using value_type = VT;
						using container_type = CT;

						static_assert(std::is_same<typename CT::value_type, VT>::value, "Value type of container must match the value type");
						
						/// <summary>
						/// Constructor for nullptr
						/// </summary>
						pointer_wrapper()
						{
							this->pointer = nullptr;
						}

						/// <summary>
						/// Constructor from pointer (does not assume ownership)
						/// </summary>
						/// <param name="pointer">Pointer</param>
						pointer_wrapper(const VT* pointer)
						{
							this->pointer = pointer;
						}

						/// <summary>
						/// Construct from moved container (destroy on desctruction)
						/// </summary>
						/// <param name="container">Container</param>
						pointer_wrapper(CT&& container)
						{
							std::swap(this->container, container);
							this->pointer = this->container.data();
						}

						/// <summary>
						/// Move operator
						/// </summary>
						/// <param name="original">Source</param>
						/// <returns>This</returns>
						pointer_wrapper& operator=(pointer_wrapper&& original)
						{
							std::swap(this->container, original.container);
							std::swap(this->pointer, original.pointer);

							return *this;
						}

						/// <summary>
						/// Return data pointer
						/// </summary>
						/// <returns>Data pointer</returns>
						const VT* get() const
						{
							return this->pointer;
						}

					private:
						/// Store pointer and container
						const VT* pointer;
						const CT container;
					};

					/// <summary>
					/// Convert original data to new type
					/// </summary>
					/// <param name="original">Pointer to original data</param>
					/// <param name="length">Length of the array</param>
					/// <typeparam name="NT">Type to convert to</typeparam>
					/// <typeparam name="OT">Type to convert from</typeparam>
					/// <returns>Wrapper around converted data</returns>
					template <typename NT, typename OT>
					pointer_wrapper<NT, std::vector<NT>> convert_if_necessary(const OT* original, std::size_t length)
					{
						if (std::is_same<NT, OT>::value)
						{
							return pointer_wrapper<NT, std::vector<NT>>(original);
						}
						else
						{
							std::vector<NT> converted(length);
							std::transform(std::begin(original, original + length, converted.begin(), [](const OT& value) { return static_cast<NT>(value); });

							return pointer_wrapper<NT, std::vector<NT>>(converted);
						}
					}

					/// <summary>
					/// Write a binary file
					/// </summary>
					/// <param name="filename">File name of the STL file</param>
					/// <param name="num_triangles">Number of triangles</param>
					/// <param name="vertices">Vertex pointer</param>
					/// <param name="normals">Normal pointer</param>
					/// <param name="indices">Index pointer</param>
					/// <typeparam name="IT">Integer type of the indices</typeparam>
					template <typename IT>
					void write_binary(const std::string& filename, uint32_t num_triangles, const float* vertices, const float* normals, const IT* indices) const;

					/// <summary>
					/// Write an ASCII file
					/// </summary>
					/// <param name="filename">File name of the STL file</param>
					/// <param name="num_triangles">Number of triangles</param>
					/// <param name="vertices">Vertex pointer</param>
					/// <param name="normals">Normal pointer</param>
					/// <param name="indices">Index pointer</param>
					/// <typeparam name="VFT">Floating point type of the vertices</typeparam>
					/// <typeparam name="NFT">Floating point type of the normals</typeparam>
					/// <typeparam name="IT">Integer type of the indices</typeparam>
					template <typename VFT, typename NFT, typename IT>
					void write_ascii(const std::string& filename, std::size_t num_triangles, const VFT* vertices, const NFT* normals, const IT* indices) const;

					/// File name
					core::param::ParamSlot filename_slot;

					/// Option for ASCII/binary
					core::param::ParamSlot ascii_binary_slot;

					/// Input
					core::CallerSlot mesh_input_slot;
				};
			}
		}
	}
}

#endif // !MEGAMOL_DATATOOLS_IO_STLWRITER_H_INCLUDED