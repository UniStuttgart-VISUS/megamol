/*
 * AbstractSTLWriter.cpp
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractSTLWriter.h"

#include "mmcore/Module.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"

namespace megamol
{
	namespace stdplugin
	{
		namespace datatools
		{
			namespace io
			{
				AbstractSTLWriter::AbstractSTLWriter() : core::Module()
					, filename_slot("STL file", "The name of to the STL file to write")
					, ascii_binary_slot("Output type", "Write an ASCII or binary file?")
				{
					// Create file name textbox
					this->filename_slot << new core::param::FilePathParam("");
					this->MakeSlotAvailable(&this->filename_slot);

					// Create enum for ASCII/binary option
					this->ascii_binary_slot << new core::param::EnumParam(0);
					this->ascii_binary_slot.Param<core::param::EnumParam>()->SetTypePair(0, "Binary");
					this->ascii_binary_slot.Param<core::param::EnumParam>()->SetTypePair(1, "ASCII");
					this->MakeSlotAvailable(&this->ascii_binary_slot);
				}

				AbstractSTLWriter::~AbstractSTLWriter()
				{ }
			}
		}
	}
}