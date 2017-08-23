/*
 * RamachandranPlot.cpp
 *
 * Author: Karsten Schatz
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */
#include "stdafx.h"
#include "RamachandranPlot.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include <iostream>

using namespace megamol;
using namespace megamol::protein_uncertainty;
using namespace megamol::protein_calls;

/**
 * RamachandranPlot::RamachandranPlot
 */
RamachandranPlot::RamachandranPlot(void) : Renderer2DModule(),
	molDataSlot("molecularDataIn","Slot for the incoming molecular data"),
	pointSize("pointSize", "The size of the drawn points"),
	ownBBParam("drawBBExtra", "Tell the renderer to draw the bounding box again in black, in case the background is white"),
	pointColorParam("pointColor", "The color of the drawn points")
#ifndef USE_SIMPLER_FONT
	,theFont(vislib::graphics::gl::FontInfo_Verdana)
#endif
{

	this->molDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(&this->molDataSlot);

	this->pointSize << new core::param::IntParam(10, 1, 20);
	this->MakeSlotAvailable(&this->pointSize);

	this->pointColorParam << new core::param::StringParam("#ffffff");
	this->MakeSlotAvailable(&this->pointColorParam);

	this->ownBBParam << new core::param::BoolParam(false);
	this->MakeSlotAvailable(&this->ownBBParam);
}

/**
 * RamachandranPlot::~RamachandranPlot
 */
RamachandranPlot::~RamachandranPlot(void) {
	Release();
}

/**
 * RamachandranPlot::create
 */
bool RamachandranPlot::create(void) {
	// be careful here, the point order is important to be able to draw all of this as GL_TRIANGLE_FAN

	// light sheets
	semiSheetPolygons.clear();
	// lower left
	std::vector<vislib::math::Vector<float, 2>> poly;
	poly.push_back(vislib::math::Vector<float, 2>(-180.0f, -163.8f));
	poly.push_back(vislib::math::Vector<float, 2>(-75.6f, -163.8f));
	poly.push_back(vislib::math::Vector<float, 2>(-46.9f, -180.0f));
	poly.push_back(vislib::math::Vector<float, 2>(-180.0f, -180.0f));
	semiSheetPolygons.push_back(poly);

	// upper left light
	poly.clear();
	poly.push_back(vislib::math::Vector<float, 2>(-180.0f, 42.9f));
	poly.push_back(vislib::math::Vector<float, 2>(-140.8f, 16.1f));
	poly.push_back(vislib::math::Vector<float, 2>(-86.0f, 16.1f));
	poly.push_back(vislib::math::Vector<float, 2>(-74.3f, 45.6f));
	poly.push_back(vislib::math::Vector<float, 2>(-74.3f, 72.5f));
	poly.push_back(vislib::math::Vector<float, 2>(-44.3f, 102.0f));
	poly.push_back(vislib::math::Vector<float, 2>(-44.3f, 161.1f));
	poly.push_back(vislib::math::Vector<float, 2>(-46.9f, 179.9f));
	poly.push_back(vislib::math::Vector<float, 2>(-180.0f, 180.0f));
	semiSheetPolygons.push_back(poly);

	// sheet central
	sureSheetPolygons.clear();
	poly.clear();
	poly.push_back(vislib::math::Vector<float, 2>(-70.4f, 91.3f));
	poly.push_back(vislib::math::Vector<float, 2>(-54.7f, 112.8f));
	poly.push_back(vislib::math::Vector<float, 2>(-54.7f, 173.2f));
	poly.push_back(vislib::math::Vector<float, 2>(-136.9f, 173.2f));
	poly.push_back(vislib::math::Vector<float, 2>(-136.9f, 155.8f));
	poly.push_back(vislib::math::Vector<float, 2>(-156.5f, 135.6f));
	poly.push_back(vislib::math::Vector<float, 2>(-156.5f, 91.3f));
	sureSheetPolygons.push_back(poly);

	// light helices
	semiHelixPolygons.clear();
	// upper right (left-handed helices)
	poly.clear();
	poly.push_back(vislib::math::Vector<float, 2>(62.6f, 14.7f));
	poly.push_back(vislib::math::Vector<float, 2>(62.6f, 96.7f));
	poly.push_back(vislib::math::Vector<float, 2>(45.6f, 79.2f));
	poly.push_back(vislib::math::Vector<float, 2>(45.6f, 26.8f));
	semiHelixPolygons.push_back(poly);

	// central left (this polygon cannot be drawn as single triangle fan)
	poly.clear();
	poly.push_back(vislib::math::Vector<float, 2>(-180.0f, -71.1f));
	poly.push_back(vislib::math::Vector<float, 2>(-180.0f, -34.9f));
	poly.push_back(vislib::math::Vector<float, 2>(-164.3f, -42.9f));
	poly.push_back(vislib::math::Vector<float, 2>(-133.0f, -42.9f));
	poly.push_back(vislib::math::Vector<float, 2>(-109.4f, -32.2f));
	poly.push_back(vislib::math::Vector<float, 2>(-106.9f, -21.4f));
	poly.push_back(vislib::math::Vector<float, 2>(-44.3f, -21.4f));
	poly.push_back(vislib::math::Vector<float, 2>(-44.3f, -71.1f));

	semiHelixPolygons.push_back(poly);

	// helix central
	sureHelixPolygons.clear();
	poly.clear();
	poly.push_back(vislib::math::Vector<float, 2>(-54.7f, -60.4f));
	poly.push_back(vislib::math::Vector<float, 2>(-54.7f, -40.2f));
	poly.push_back(vislib::math::Vector<float, 2>(-100.4f, -40.2f));
	poly.push_back(vislib::math::Vector<float, 2>(-123.9f, -51.0f));
	poly.push_back(vislib::math::Vector<float, 2>(-156.5f, -51.0f));
	poly.push_back(vislib::math::Vector<float, 2>(-156.5f, -60.4f));
	sureHelixPolygons.push_back(poly);
	return true;
}

/**
 * RamachandranPlot::release
 */
void RamachandranPlot::release(void) {
}

/**
 * RamachandranPlot::MouseEvent
 */
bool RamachandranPlot::MouseEvent(float x, float y, core::view::MouseFlags flags) {
	bool consumeEvent = false;
	return consumeEvent;
}

/**
 * RamachandranPlot::GetExtents
 */
bool RamachandranPlot::GetExtents(core::view::CallRender2D& call) {
	call.SetBoundingBox(-180.0f, -180.0f, 180.0f, 180.0f);

	MolecularDataCall * mol = this->molDataSlot.CallAs<MolecularDataCall>();

	if (mol != nullptr && ((*mol)(MolecularDataCall::CallForGetExtent))) {
		call.SetTimeFramesCount(mol->FrameCount());
	} else {
		call.SetTimeFramesCount(1);
	}

	return true;
}

/**
 * RamachandranPlot::Render
 */
bool RamachandranPlot::Render(core::view::CallRender2D& call) {

	MolecularDataCall * mol = this->molDataSlot.CallAs<MolecularDataCall>();
	if (mol == nullptr) return false;

	mol->SetFrameID(static_cast<unsigned int>(call.Time()), true);
	if (!(*mol)(MolecularDataCall::CallForGetExtent)) return false;
	mol->SetFrameID(static_cast<unsigned int>(call.Time()), true);
	if (!(*mol)(MolecularDataCall::CallForGetData)) return false;

	computeDihedralAngles(mol);
	computePolygonPositions();

	// draw the unsure sheet polygons
	for (auto & poly : semiSheetPolygons) {
		glBegin(GL_TRIANGLE_FAN);
		glColor3f(0.5f * 0.1216f, 0.5f * 0.470588f, 0.5f * 0.70588f);
		for (auto & vertex : poly) {
			glVertex2f(vertex.X(), vertex.Y());
		}
		glEnd();
	}

	// draw the sure sheet polygons
	for (auto & poly : sureSheetPolygons) {
		glBegin(GL_TRIANGLE_FAN);
		glColor3f(0.1216f, 0.470588f, 0.70588f);
		for (auto & vertex : poly) {
			glVertex2f(vertex.X(), vertex.Y());
		}
		glEnd();
	}

	// draw the unsure helix polygons
	// BE CAREFUL HERE
	// The second drawn polygon is not convex and requires a splitting of the 
	// rendered area
	int i = 0;
	for (auto & poly : semiHelixPolygons) {
		glBegin(GL_TRIANGLE_FAN);
		glColor3f(0.5f * 0.890196f, 0.5f * 0.10196f, 0.5f * 0.109804f);
		int j = 0;
		for (auto & vertex : poly) {
			glVertex2f(vertex.X(), vertex.Y());
			if (i == 1 && j == 4) {
				glVertex2f(poly[7].X(), poly[7].Y());
				glEnd();
				glBegin(GL_TRIANGLE_FAN);
				glColor3f(0.5f * 0.890196f, 0.5f * 0.10196f, 0.5f * 0.109804f);
				glVertex2f(vertex.X(), vertex.Y());
			}
			j++;
		}
		glEnd();
		i++;
	}

	// draw the sure helix polygons
	for (auto & poly : sureHelixPolygons) {
		glBegin(GL_TRIANGLE_FAN);
		glColor3f(0.890196f, 0.10196f, 0.109804f);
		for (auto & vertex : poly) {
			glVertex2f(vertex.X(), vertex.Y());
		}
		glEnd();
	}

	glEnable(GL_LINE_STIPPLE);
	glLineStipple(1, 0x5555);
	glBegin(GL_LINES);
	glColor3f(0.4f, 0.4f, 0.4f);
	glVertex2f(0.0f, -180.0f);
	glVertex2f(0.0f, 180.0f);
	glVertex2f(-180.0f, 0.0f);
	glVertex2f(180.0f, 0.0f);
	glEnd();
	glDisable(GL_LINE_STIPPLE);


	float r, g, b;
	core::utility::ColourParser::FromString(this->pointColorParam.Param<core::param::StringParam>()->Value(), r, g, b);
	glPointSize(static_cast<GLfloat>(this->pointSize.Param<core::param::IntParam>()->Value()));
	// draw the points
	glBegin(GL_POINTS);
	glColor3f(r, g, b);
	for (int i = 0; i < static_cast<int>(this->angles.size()); i++) {
		for (int j = 0; j < static_cast<int>(this->angles[i].size()); j = j + 2) {
			switch (this->pointStates[i][j / 2]) {
			case NONE: 
				glColor3f(r, g, b);
				break;
			case UNSURE_ALPHA:
				glColor3f(0.5f, 0.5f, 0.0f);
				break;
			case SURE_ALPHA:
				glColor3f(1.0f, 1.0f, 0.0f);
				break;
			case UNSURE_BETA:
				glColor3f(0.0f, 0.5f, 0.5f);
				break;
			case SURE_BETA:
				glColor3f(0.0f, 1.0f, 1.0f);
				break;
			default: 
				glColor3f(r, g, b);
				break;
			}
			if (this->angles[i][j] > -500.0f && this->angles[i][j + 1] > -500.0f) {
				glVertex2f(this->angles[i][j], this->angles[i][j + 1]);
			}
		}
	}
	glEnd();

	if (this->ownBBParam.Param<core::param::BoolParam>()->Value()) {
		// draw the bounding box
		glLineWidth(1.0f);
		glBegin(GL_LINE_LOOP);
		glColor3f(0.0f, 0.0f, 0.0f);
		glVertex2f(-180.0f, -180.0f);
		glVertex2f(-180.0f, 180.0f);
		glVertex2f(180.0f, 180.0f);
		glVertex2f(180.0f, -180.0f);
		glEnd();
	}

	if (theFont.Initialise()) {
		vislib::StringA mystring;
		float fontsize = 20.0f;
		float wordlength;

		// draw labels
		mystring = L"phi";
		wordlength = theFont.LineWidth(fontsize, mystring);
		theFont.DrawString(0.0, -200.0f, wordlength + 1.0f, 1.0f, mystring, vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);

		mystring = L"psi";
		wordlength = theFont.LineWidth(fontsize, mystring);
		theFont.DrawString(-200.0f, 0.0f, wordlength + 1.0f, 1.0f, mystring, vislib::graphics::AbstractFont::ALIGN_RIGHT_MIDDLE);

		fontsize = 10.0f;

		mystring = L"0°";
		wordlength = theFont.LineWidth(fontsize, mystring);
		theFont.DrawString(0.0f, -180.0f, wordlength + 1.0f, 1.0f, mystring, vislib::graphics::AbstractFont::ALIGN_CENTER_TOP);
		theFont.DrawString(-180.0f, 0.0f, wordlength + 1.0f, 1.0f, mystring, vislib::graphics::AbstractFont::ALIGN_RIGHT_MIDDLE);

		fontsize = 5.0f;
		mystring = L"180°";
		wordlength = theFont.LineWidth(fontsize, mystring);
		theFont.DrawString(180.0f, -180.0f, wordlength + 1.0f, 1.0f, mystring, vislib::graphics::AbstractFont::ALIGN_RIGHT_TOP);
		theFont.DrawString(-180.0f, 180.0f, wordlength + 1.0f, 1.0f, mystring, vislib::graphics::AbstractFont::ALIGN_RIGHT_TOP);

		mystring = L"-180°";
		wordlength = theFont.LineWidth(fontsize, mystring);
		theFont.DrawString(-180.0f, -180.0f, wordlength + 1.0f, 1.0f, mystring, vislib::graphics::AbstractFont::ALIGN_RIGHT_BOTTOM);
		theFont.DrawString(-180.0f, -180.0f, wordlength + 1.0f, 1.0f, mystring, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
	}

	return true;
}

/**
 * RamachandranPlot::computeDihedralAngles
 */
void RamachandranPlot::computeDihedralAngles(MolecularDataCall * mol) {
	this->angles.clear();

	unsigned int firstResIdx = 0;
	unsigned int lastResIdx = 0;
	unsigned int firstAtomIdx = 0;
	unsigned int lastAtomIdx = 0;
	unsigned int atomTypeIdx = 0;
	unsigned int firstSecIdx = 0;
	unsigned int lastSecIdx = 0;
	unsigned int firstAAIdx = 0;
	unsigned int lastAAIdx = 0;

	int molCount = mol->MoleculeCount();
	this->angles.resize(molCount);
	this->pointStates.resize(molCount);

	// the 5 needed positions for the dihedral angles
	vislib::math::Vector<float, 3> prevCPos;
	vislib::math::Vector<float, 3> NPos;
	vislib::math::Vector<float, 3> CaPos;
	vislib::math::Vector<float, 3> CPos;
	vislib::math::Vector<float, 3> nextNPos;

	for (int molIdx = 0; molIdx < molCount; molIdx++) {
		MolecularDataCall::Molecule chain = mol->Molecules()[molIdx];

		if (mol->Residues()[chain.FirstResidueIndex()]->Identifier() != MolecularDataCall::Residue::AMINOACID) {
			continue;
		}

		firstSecIdx = chain.FirstSecStructIndex();
		lastSecIdx = firstSecIdx + chain.SecStructCount();

		for (unsigned int secIdx = firstSecIdx; secIdx < lastSecIdx; secIdx++) {
			firstAAIdx = mol->SecondaryStructures()[secIdx].FirstAminoAcidIndex();
			lastAAIdx = firstAAIdx + mol->SecondaryStructures()[secIdx].AminoAcidCount();

			MolecularDataCall::SecStructure secStruct = mol->SecondaryStructures()[secIdx];

			for (unsigned int aaIdx = firstAAIdx; aaIdx < lastAAIdx; aaIdx++) {

				MolecularDataCall::AminoAcid * acid = nullptr;

				if (mol->Residues()[aaIdx]->Identifier() == MolecularDataCall::Residue::AMINOACID) {
					acid = (MolecularDataCall::AminoAcid*)(mol->Residues()[aaIdx]);
				} else {
					// TODO is this correct?
					this->angles[molIdx].push_back(-1000.0f);
					this->angles[molIdx].push_back(-1000.0f);
					this->pointStates[molIdx].push_back(PointState::NONE);
					continue;
				}

				float phi = 0.0f;
				float psi = 0.0f;

				// get all relevant atom positions of the current amino acid
				NPos = vislib::math::Vector<float, 3>(&mol->AtomPositions()[acid->NIndex() * 3]);
				CaPos = vislib::math::Vector<float, 3>(&mol->AtomPositions()[acid->CAlphaIndex() * 3]);
				CPos = vislib::math::Vector<float, 3>(&mol->AtomPositions()[acid->CCarbIndex() * 3]);

				// get all relevant atom positions of the last and the next amino acid
				
				// is this the first residue?
				if (aaIdx == chain.FirstResidueIndex()) {
					// we have no prevCPos
					phi = -1000.0f;
				} else {
					acid = (MolecularDataCall::AminoAcid*)(mol->Residues()[aaIdx - 1]);
					if (acid != nullptr) {
						prevCPos = vislib::math::Vector<float, 3>(&mol->AtomPositions()[acid->CCarbIndex() * 3]);
					} else {
						phi = -1000.0f;
					}
				}

				if (aaIdx == chain.FirstResidueIndex() + chain.ResidueCount() - 1) {
					// we have no nextNPos
					psi = -1000.0f;
				} else {
					acid = (MolecularDataCall::AminoAcid*)(mol->Residues()[aaIdx + 1]);
					if (acid != nullptr) {
						nextNPos = vislib::math::Vector<float, 3>(&mol->AtomPositions()[acid->NIndex() * 3]);
					} else {
						psi = -1000.0f;
					}
				}

				// if nothing speaks against it, compute the angles
				if (phi > -500.0f) {
					phi = dihedralAngle(prevCPos, NPos, CaPos, CPos);
				}
				if (psi > -500.0f) {
					psi = dihedralAngle(NPos, CaPos, CPos, nextNPos);
				}

				this->angles[molIdx].push_back(phi);
				this->angles[molIdx].push_back(psi);
				this->pointStates[molIdx].push_back(PointState::NONE);

				/*acid = (MolecularDataCall::AminoAcid*)(mol->Residues()[aaIdx]);
				if (acid != nullptr) {
					std::cout << "Acid " << aaIdx << ": " << mol->ResidueTypeNames()[acid->Type()] << "; phi = " << phi << "  psi = " << psi << std::endl;
				}*/
			}
		}
	}
}

/**
 * RamachandranPlot::computePolygonPositions
 */
void RamachandranPlot::computePolygonPositions(void) {
	for (int molIdx = 0; molIdx < static_cast<int>(this->angles.size()); molIdx++) {
		for (int i = 0; i < static_cast<int>(this->angles[molIdx].size()); i = i + 2) {
			vislib::math::Vector<float, 2> pos(&this->angles[molIdx][i]);

			if (pos.X() < -500.0f || pos.Y() < -500.0f) continue;

			float posx = pos.X();
			float posy = pos.Y();

			bool semiHelixState = false;
			bool semiSheetState = false;
			bool trueHelixState = false;
			bool trueSheetState = false;

			// test against sheets
			for (auto & semiSheet : this->semiSheetPolygons) {
				semiSheetState = locateInPolygon(semiSheet, pos) ? true : semiSheetState;
			}

			for (auto & sureSheet : this->sureSheetPolygons) {
				trueSheetState = locateInPolygon(sureSheet, pos) ? true : trueSheetState;
			}

			// test against helices
			for (auto & semiHelix : this->semiHelixPolygons) {
				semiHelixState = locateInPolygon(semiHelix, pos) ? true : semiHelixState;
			}

			for (auto & sureHelix : this->sureHelixPolygons) {
				trueHelixState = locateInPolygon(sureHelix, pos) ? true : trueHelixState;
			}

			if (semiSheetState) {
				this->pointStates[molIdx][i / 2] = PointState::UNSURE_BETA;
			}
			if (trueSheetState) {
				this->pointStates[molIdx][i / 2] = PointState::SURE_BETA;
			}
			if (semiHelixState) {
				this->pointStates[molIdx][i / 2] = PointState::UNSURE_ALPHA;
			}
			if (trueHelixState) {
				this->pointStates[molIdx][i / 2] = PointState::SURE_ALPHA;
			}
		}
	}
}

/**
 * RamachandranPlot::dihedralAngle
 */
float RamachandranPlot::dihedralAngle(const vislib::math::Vector<float, 3>& v1, const vislib::math::Vector<float, 3>& v2, 
	const vislib::math::Vector<float, 3>& v3, const vislib::math::Vector<float, 3>& v4) {

	/* 
	 *	Code from https://github.com/biopython/biopython/blob/9fa26d5efb38e7398f82f4b4e028ca84b7eeaa58/Bio/PDB/Vector.py
	 */
	auto ab = v1 - v2;
	auto cb = v3 - v2;
	auto db = v4 - v3;
	auto u = ab.Cross(cb);
	auto v = db.Cross(cb);
	auto w = u.Cross(v);
	float result = u.Angle(v);
	if (cb.Angle(w) > 0.001f) {
		result = -result;
	}
	return vislib::math::AngleRad2Deg(result);
}

/**
 * RamachandranPlot::locateInPolygon
 */
bool RamachandranPlot::locateInPolygon(const std::vector<vislib::math::Vector<float, 2>>& polyVector, const vislib::math::Vector<float, 2> inputPos) {
	int nvert = static_cast<int>(polyVector.size());
	int i, j, c = 0;
	float testx = inputPos.X();
	float testy = inputPos.Y();
	for (i = 0, j = nvert - 1; i < nvert; j = i++) {
		float vertxi = polyVector[i].X();
		float vertxj = polyVector[j].X();
		float vertyi = polyVector[i].Y();
		float vertyj = polyVector[j].Y();

		if (((vertyi > testy) != (vertyj > testy)) &&
			(testx < (vertxj - vertxi) * (testy - vertyi) / (vertyj - vertyi) + vertxi)) {
			c = !c;
		}
	}
	return (c != 0);
}