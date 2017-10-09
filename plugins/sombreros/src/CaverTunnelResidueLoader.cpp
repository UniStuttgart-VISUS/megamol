/*
 * CaverTunnelResidueLoader.cpp
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "CaverTunnelResidueLoader.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::sombreros;

/*
 * CaverTunnelResidueLoader::CaverTunnelResidueLoader
 */
CaverTunnelResidueLoader::CaverTunnelResidueLoader(void) : view::AnimDataModule(),
		getData("getData", "The slot providing the data loaded by this module."),
		filenameSlot("filename", "The path to the input file.") {

	// TODO
}

/*
 * CaverTunnelResidueLoader::~CaverTunnelResidueLoader
 */
CaverTunnelResidueLoader::~CaverTunnelResidueLoader(void) {
	this->Release();
}

/*
 * CaverTunnelResidueLoader::constructFrame
 */
core::view::AnimDataModule::Frame* CaverTunnelResidueLoader::constructFrame(void) const {
	// TODO
	return nullptr;
}

/*
 * CaverTunnelResidueLoader::create
 */
bool CaverTunnelResidueLoader::create(void) {
	// TODO
	return true;
}

/*
 * CaverTunnelResidueLoader::loadFrame
 */
void CaverTunnelResidueLoader::loadFrame(core::view::AnimDataModule::Frame * frame, unsigned int idx) {
	// TODO
}

/*
 * CaverTunnelResidueLoader::release
 */
void CaverTunnelResidueLoader::release(void) {
	// TODO
}

/*
 * CaverTunnelResidueLoader::filenameChanged
 */
bool CaverTunnelResidueLoader::filenameChanged(core::param::ParamSlot& slot) {
	// TODO
	return true;
}

/*
 * CaverTunnelResidueLoader::getDataCallback
 */
bool CaverTunnelResidueLoader::getDataCallback(core::Call& caller) {
	// TODO
	return true;
}

/*
 * CaverTunnelResidueLoader::getExtentCallback
 */
bool CaverTunnelResidueLoader::getExtentCallback(core::Call& caller) {
	// TODO
	return true;
}