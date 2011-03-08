#include "stdafx.h"
#include "MoleculeSequenceRenderer.h"
#include "MolecularDataCall.h"
#include "CallerSlot.h"
#include "vislib/SimpleFont.h"

//#include "vislib/

using namespace megamol::core::view;
using namespace megamol::core;
using namespace megamol::protein;

MoleculeSequenceRenderer::MoleculeSequenceRenderer(void) : 
	dataCall(new CallerSlot("getdata", "call used to fetch data from a data provider"))
{
	this->dataCall->SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable(this->dataCall);
}

MoleculeSequenceRenderer::~MoleculeSequenceRenderer(void)
{
	delete this->dataCall;
	this->Release();
}

bool MoleculeSequenceRenderer::create(void)
{
	return true;
}

void MoleculeSequenceRenderer::release(void)
{
}

bool MoleculeSequenceRenderer::GetExtents(CallRender2D& call)
{
	MolecularDataCall* data = dataCall->CallAs<MolecularDataCall>();
	if(!data) return false;
	(*data)(MolecularDataCall::CallForGetData);

	unsigned int residueCount = data->ResidueCount();
	
	call.SetBoundingBox(0, 0, residueCount * 20, 45);



	return true;
}

bool MoleculeSequenceRenderer::Render(CallRender2D& call)
{
	widgetLibrary.clear();

	MolecularDataCall* data = dataCall->CallAs<MolecularDataCall>();
	if(!data) return false;
	
	(*data)(MolecularDataCall::CallForGetData);
	
	unsigned int chainCount = data->ChainCount();
	const MolecularDataCall::Chain *chains = data->Chains();
	unsigned int moleculeCount;// = data->MoleculeCount();
	const MolecularDataCall::Molecule *molecules = data->Molecules();
	unsigned int residueCount;// = data->ResidueCount();
	const MolecularDataCall::Residue **residues = data->Residues();

	const float buttonWidth = 20;
	const float buttonHeight = 10;
	const float buttonSpacing = 2;

	unsigned int moleculePosition = 0;
	unsigned int residuePosition = 0;
	unsigned int chainPosition = 0;
	unsigned int residuesWidth = 0;
	unsigned int moleculeWidth = 0;

	unsigned int firstMolecule;
	unsigned int firstResidue;

    vislib::StringA tmpStr;

	for(unsigned int chain = 0; chain < chainCount; ++chain)
	{
		moleculeCount = chains[chain].MoleculeCount();
		firstMolecule = chains[chain].FirstMoleculeIndex();
		residueCount = 0;
		moleculeWidth = 0;
		for(unsigned int molecule = firstMolecule; molecule < firstMolecule + moleculeCount; ++molecule)
		{
			residueCount = molecules[molecule].ResidueCount();
			firstResidue = molecules[molecule].FirstResidueIndex();
			residuesWidth = 0;
			for(unsigned int residue = firstResidue; residue < firstResidue + residueCount; ++residue)
			{
				//TODO: filter AminoAcids
				this->paintButton(
						residuePosition, 0, 
						buttonWidth - buttonSpacing, buttonHeight, 
						0.4f, 0.4f, 0.6f,
						//"residue"
                        data->ResidueTypeNames()[residues[residue]->Type()].PeekBuffer()
				);
				residuePosition += buttonWidth;
			}

			residuesWidth = residueCount * (buttonWidth + buttonSpacing);		
			
            tmpStr.Format( " Molecule %i", molecule);
			this->paintButton(
					moleculePosition, 15,
					residuesWidth - buttonSpacing, buttonHeight, 
					1.0f, 0.8f, 0.0f,
					//"Molecule"
                    tmpStr.PeekBuffer()
			);
			moleculePosition += residuesWidth;
		}
		
		moleculeWidth = moleculeCount * (residuesWidth + buttonSpacing);

        tmpStr.Format( " Chain %i", chain);
		this->paintButton(
				chainPosition, 30,
				moleculeWidth, buttonHeight, 
				1.0f, 0.0f, 0.0f,
				//"chain"
                tmpStr
		);
		chainPosition += moleculeWidth;
	}

	
	widgetLibrary.renderWidgets();
	return true;
}

void MoleculeSequenceRenderer::paintButton(float x, float y, float w, float h, float r, float g, float b, const char* text)
{
	using namespace vislib::graphics::gl;
	glBegin(GL_POLYGON);
	glColor3f(r, g, b); 

	glVertex2f(x,	y);
	glVertex2f(x+w,	y);
	glVertex2f(x+w,	y+h);
	glVertex2f(x,	y+h);

	glEnd();
    
	SimpleFont f;
    if( f.Initialise() ) {
	    glColor3f( 1.0f, 1.0f, 1.0f); 
	    f.DrawString( x, y+h, int( h), true, text);
    }
}

#if 1
bool MoleculeSequenceRenderer::MouseEvent(float x, float y, MouseFlags flags)
{
	widgetLibrary.mouseHandler(x, y, flags);
	return false;
}

#endif