/*
 * Diagram2DRenderer.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All Rights reserved.
 */

#include "stdafx.h"

#define _USE_MATH_DEFINES 1

#include "Diagram2DRenderer.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/utility/ColourParser.h"
#include "vislib/graphics/gl/SimpleFont.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>
#include <math.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/*
 * Diagram2DRenderer::Diagram2DRenderer (CTOR)
 */
Diagram2DRenderer::Diagram2DRenderer( void ) : Renderer2DModule (),
        dataCallerSlot( "getData", "Connects the diagram rendering with data storage." ), 
        resolutionParam( "resolution", "The plotting resolution of the diagram."),
        plotColorParam( "plotcolor", "The color used for plotting the diagram."),
        clearDiagramParam( "clearDiagram", "Clears the diagram"),
        currentFbo( 0), oldDataPoint( 0.0f, 0.0f) {
    // segmentation data caller slot
    this->dataCallerSlot.SetCompatibleCall<Diagram2DCallDescription>();
    this->MakeSlotAvailable( &this->dataCallerSlot);

    // set up the resolution param for the texture
    this->resolutionParam.SetParameter( new param::IntParam( 1024, 0, 8192) );
    this->MakeSlotAvailable( &this->resolutionParam);

    // set up the plot color param
    this->plotColorParam.SetParameter( new param::StringParam( "#bb0000"));
    this->MakeSlotAvailable( &this->plotColorParam);
    
    // set up the clear diagram param
    this->clearDiagramParam.SetParameter( new param::ButtonParam(core::view::Key::KEY_DELETE));
    this->MakeSlotAvailable( &this->clearDiagramParam);

    // set the label space
    this->labelSpace.Set( -1.0f, 1.0f,-1.0);
}

/*
 * Diagram2DRenderer::~Diagram2DRenderer (DTOR)
 */
Diagram2DRenderer::~Diagram2DRenderer( void ) {
    this->Release();
}

/*
 * Diagram2DRenderer::create
 */
bool Diagram2DRenderer::create() {

    return true;
}

/*
 * Diagram2DRenderer::release
 */
void Diagram2DRenderer::release() {
    // release fbos
    this->fbo[0].Release();
    this->fbo[1].Release();
}

bool Diagram2DRenderer::GetExtents( view::CallRender2DGL& call) {
    // set the bounding box to 0..1
    call.AccessBoundingBoxes().SetBoundingBox( 0.0f, 0.0f, 0, 1.0f, 1.0f, 0);

    return true;
}


/*
 * Callback for mouse events (move, press, and release)
 */
bool Diagram2DRenderer::MouseEvent(float x, float y, view::MouseFlags flags) {
    bool consumeEvent = false;

    return consumeEvent;
}

/*
 * Diagram2DRenderer::Render
 */
bool Diagram2DRenderer::Render( view::CallRender2DGL &call) {
    // get pointer to Diagram2DCall
    Diagram2DCall *diagram = this->dataCallerSlot.CallAs<Diagram2DCall>();
    if( diagram == NULL ) return false;
    // execute the call
    if( !(*diagram)(Diagram2DCall::CallForGetData) ) return false;

    // get the new data point
    vislib::math::Vector<float, 2> dataPoint( diagram->GetValuePair());
    // normalize point to -1..1
    dataPoint.SetX( dataPoint.X() / diagram->GetRangeX());
    dataPoint.SetY( dataPoint.Y() / diagram->GetRangeY());
    dataPoint *= 2.0f;
    dataPoint -= vislib::math::Vector<float, 2>( 1.0f, 1.0f);

    // refresh parameters
    this->parameterRefresh();

    // generate and clear the FBOs, if necessary
    this->generateDiagramTextures();

    // store old Fbo
    unsigned int oldFbo = this->currentFbo;
    // change fbo
    this->currentFbo = ( this->currentFbo + 1) % 2;

    // set clear FBOs (white)
    float bgColor[4];
    glGetFloatv( GL_COLOR_CLEAR_VALUE, bgColor);
    glClearColor( 1.0f, 1.0f, 1.0f, 1.0f);
    // enable render to texture
    this->fbo[this->currentFbo].Enable();
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // set OGL point and line parameters
    glPointSize( 3.0f);
    glHint( GL_POINT_SMOOTH, GL_NICEST);
    glEnable( GL_POINT_SMOOTH);
    glEnable( GL_POINT_SIZE);
    glLineWidth( 3.0f);
    glHint( GL_LINE_SMOOTH, GL_NICEST);
    glEnable( GL_LINE_SMOOTH);
    glEnable( GL_LINE_WIDTH);

    // draw the old diagram texture
    this->fbo[oldFbo].DrawColourTexture();

    ::glMatrixMode(GL_PROJECTION);
    ::glPushMatrix();
    ::glLoadIdentity();

    ::glMatrixMode(GL_MODELVIEW);
    ::glPushMatrix();
    ::glLoadIdentity();

    float s =  float( resolutionParam.Param<param::IntParam>()->Value()) / 10000.0f;

    // draw marker, if requested
    glEnable( GL_LINE_STIPPLE);
    glLineStipple( 2, 0x00FF);
    if( diagram->Marker() ) {
        vislib::graphics::gl::SimpleFont f;
        vislib::StringA tmpStr;
        if( f.Initialise() ) {
            tmpStr.Format( " %.2f", diagram->GetX());
            s = float( resolutionParam.Param<param::IntParam>()->Value()) / 10000.0f;
            if( this->labelSpace.X() < dataPoint.X() || labelSpace.Z() > dataPoint.X() ) {
                glColor3f( 1.0f, 1.0f, 1.0f);
                glBegin( GL_QUADS);
                glVertex2f( dataPoint.X(), 1.0f);
                glVertex2f( dataPoint.X(), 1.0f - f.LineHeight( s));
                glVertex2f( dataPoint.X() + f.LineWidth( s, tmpStr.PeekBuffer()), 1.0f - f.LineHeight( s));
                glVertex2f( dataPoint.X() + f.LineWidth( s, tmpStr.PeekBuffer()), 1.0f);
                glEnd();
                glColor3f( 0.5f, 0.5f, 0.5f);
                f.DrawString( dataPoint.X(), 1.0f, s, true, tmpStr.PeekBuffer());
                this->labelSpace.Set( dataPoint.X() + f.LineWidth( s, tmpStr.PeekBuffer()), 1.0f - f.LineHeight( s), dataPoint.X());
            } else {
                glColor3f( 1.0f, 1.0f, 1.0f);
                glBegin( GL_QUADS);
                glVertex2f( dataPoint.X(), labelSpace.Y());
                glVertex2f( dataPoint.X(), labelSpace.Y() - f.LineHeight( s));
                glVertex2f( dataPoint.X() + f.LineWidth( s, tmpStr.PeekBuffer()), labelSpace.Y() - f.LineHeight( s));
                glVertex2f( dataPoint.X() + f.LineWidth( s, tmpStr.PeekBuffer()), labelSpace.Y());
                glEnd();
                glColor3f( 0.5f, 0.5f, 0.5f);
                f.DrawString( dataPoint.X(), this->labelSpace.Y(), s, true, tmpStr.PeekBuffer());
                this->labelSpace.Set( dataPoint.X() + f.LineWidth( s, tmpStr.PeekBuffer()), this->labelSpace.Y() - f.LineHeight( s), dataPoint.X());
            }
        }
        glColor3f( 0.5f, 0.5f, 0.5f);
        glBegin( GL_LINES);
        glVertex2f( dataPoint.X(),-1.0f);
        glVertex2f( dataPoint.X(), 1.0f);
        glEnd();
    }
    glDisable( GL_LINE_STIPPLE);

    // draw the new data point
    glColor3fv( this->plotColor.PeekComponents());
    // draw line, if the difference is small (but positive
    if( ( dataPoint.X() - this->oldDataPoint.X()) < ( 2.0f / diagram->GetRangeX()) &&
        ( dataPoint.X() - this->oldDataPoint.X()) >= 0.0f) {
        glBegin( GL_LINES);
        glVertex2fv( this->oldDataPoint.PeekComponents());
        glVertex2fv( dataPoint.PeekComponents());
        glEnd();
    } else {
        glBegin( GL_POINTS);
        glVertex2fv( dataPoint.PeekComponents());
        glEnd();
    }

    ::glPopMatrix();

    ::glMatrixMode(GL_PROJECTION);
    ::glPopMatrix();

    // reset clear color
    glClearColor( bgColor[0], bgColor[1], bgColor[2], bgColor[3]);

    // disable render to texture
    this->fbo[this->currentFbo].Disable();
    
    // draw the result
    //this->fbo[this->currentFbo].DrawColourTexture();
    glEnable( GL_TEXTURE_2D);
    glBindTexture( GL_TEXTURE_2D, this->fbo[this->currentFbo].GetColourTextureID());
    glColor3f( 1, 1, 1);
    glBegin( GL_QUADS);
    glTexCoord2f( 0, 0);
    glVertex2f( 0, 0);
    glTexCoord2f( 1, 0);
    glVertex2f( 1, 0);
    glTexCoord2f( 1, 1);
    glVertex2f( 1, 1);
    glTexCoord2f( 0, 1);
    glVertex2f( 0, 1);
    glEnd(); // GL_QUADS
    glBindTexture( GL_TEXTURE_2D, 0);
    glDisable( GL_TEXTURE_2D);

    // draw the marker for the time
    glColor3f( 1.0f, 0.75f, 0.0f);
    float ct = diagram->CallTime();
    ct /= diagram->GetRangeX();
    glEnable( GL_LINE_STIPPLE);
    glLineStipple( 2, 0x0303);
    glBegin( GL_LINES);
    glVertex2f( ct, 0.0f);
    glVertex2f( ct, 1.0f);
    glEnd(); // GL_LINES
    glDisable( GL_LINE_STIPPLE);
    vislib::StringA ctStr;
    vislib::graphics::gl::SimpleFont ctFont;
    if( ctFont.Initialise() ) {
        ctStr.Format( " %.2f", diagram->CallTime());
        s /= 2.0f;
        ctFont.DrawString( ct, 1.0f - ctFont.LineHeight( s), s, true, ctStr.PeekBuffer());
    }

    // reset OGL point parameters
    glPointSize( 1.0f);
    glDisable( GL_POINT_SMOOTH);
    glDisable( GL_POINT_SIZE);
    glLineWidth( 1.0f);
    glDisable( GL_LINE_SMOOTH);
    glDisable( GL_LINE_WIDTH);

    // store the new data point
    this->oldDataPoint = dataPoint;

    return true;
}

/*
 * refresh all parameters
 */
void Diagram2DRenderer::parameterRefresh() {
    // RGB color values
    float r, g, b;
    // get plot color
    utility::ColourParser::FromString( 
        this->plotColorParam.Param<param::StringParam>()->Value(),
        r, g, b);
    this->plotColor.Set( r, g, b);

    // clear
    if( this->clearDiagramParam.IsDirty() ) {
        this->clearDiagram();
        this->clearDiagramParam.ResetDirty();
    }
}

/*
 * generate the diagram texture and FBOs
 */
void Diagram2DRenderer::generateDiagramTextures() {
    // flag for fbo clearing
    bool clearFbo = false;
    // get the resolution
    int res = this->resolutionParam.Param<param::IntParam>()->Value();
    // create both FBOs, if necessary
    if( this->resolutionParam.IsDirty() ) {
        this->fbo[0].Create( res, res, GL_RGB, GL_RGB, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
        this->fbo[1].Create( res, res, GL_RGB, GL_RGB, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
        this->resolutionParam.ResetDirty();
        clearFbo = true;
    }
    if( !this->fbo[0].IsValid() ) {
        this->fbo[0].Create( res, res, GL_RGB, GL_RGB, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
        clearFbo = true;
    }
    if( !this->fbo[1].IsValid() ) {
        this->fbo[1].Create( res, res, GL_RGB, GL_RGB, GL_FLOAT, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
        clearFbo = true;
    }

    if( clearFbo ) {
        this->clearDiagram();
    }

}

/*
 * clears the diagram textures
 */
void Diagram2DRenderer::clearDiagram() {
    // clear both FBOs (white)
    float bgColor[4];
    glGetFloatv( GL_COLOR_CLEAR_VALUE, bgColor);
    glClearColor( 1.0f, 1.0f, 1.0f, 1.0f);
    if( this->fbo[0].IsValid() ) {
        this->fbo[0].Enable();
        glClear( GL_COLOR_BUFFER_BIT );
        this->fbo[0].Disable();
    }
    if( this->fbo[1].IsValid() ) {
        this->fbo[1].Enable();
        glClear( GL_COLOR_BUFFER_BIT );
        this->fbo[1].Disable();
    }
    glClearColor( bgColor[0], bgColor[1], bgColor[2], bgColor[3]);
    
    // set the label space
    this->labelSpace.Set( -1.0f, 1.0f,-1.0);
}
