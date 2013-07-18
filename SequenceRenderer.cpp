#include "stdafx.h"
#include "SequenceRenderer.h"

#include "SequenceRenderer.h"
#include "MolecularDataCall.h"
#include "CoreInstance.h"
#include "param/IntParam.h"
#include "utility/ColourParser.h"
#include "vislib/SimpleFont.h"
#include "vislib/Rectangle.h"
#include "vislib/BufferedFile.h"
#include "vislib/sysfunctions.h"
#include "glh/glh_extensions.h"
#include <GL/glu.h>
#include <math.h>
//#include <vislib/verdana.inc>
#include "misc/ImageViewer.h"
#include "utility/ResourceWrapper.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace vislib::graphics::gl;
using vislib::sys::Log;

/*
 * SequenceRenderer::SequenceRenderer (CTOR)
 */
SequenceRenderer::SequenceRenderer( void ) : Renderer2DModule (),
        dataCallerSlot( "getData", "Connects the diagram rendering with data storage." ),
        resCountPerRowParam( "ResPerRow", "The number of residues per row" ) {

    // segmentation data caller slot
    this->dataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    // param slot for number of residues per row
    this->resCountPerRowParam.SetParameter( new param::IntParam( 50, 10));
    this->MakeSlotAvailable( &this->resCountPerRowParam);
}

/*
 * Diagram2DRenderer::~Diagram2DRenderer (DTOR)
 */
SequenceRenderer::~SequenceRenderer( void ) {
    this->Release();
}

/*
 * Diagram2DRenderer::create
 */
bool SequenceRenderer::create() {
    
    return true;
}

/*
 * Diagram2DRenderer::release
 */
void SequenceRenderer::release() {
}

bool SequenceRenderer::GetExtents(view::CallRender2D& call) {

    return true;
}


/*
 * Callback for mouse events (move, press, and release)
 */
bool SequenceRenderer::MouseEvent(float x, float y, view::MouseFlags flags) {
    bool consumeEvent = false;


    return consumeEvent;
}


/*
 * Diagram2DRenderer::Render
 */
bool SequenceRenderer::Render(view::CallRender2D &call) {
    // get pointer to Diagram2DCall
    MolecularDataCall *mol = this->dataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL ) return false;
    
    // execute the call
    if( !(*mol)(MolecularDataCall::CallForGetData) ) return false;

    return true;
}
