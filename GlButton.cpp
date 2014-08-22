#include "stdafx.h"
#include "GlButton.h"
#include "vislib/IncludeAllGL.h"
#include <GL/glu.h>


GlButton::GlButton(void)
{
}

GlButton::~GlButton(void)
{
}

void GlButton::move(float x, float y)
{
	this->x = x;
	this->y = y;
}

void GlButton::resize(float w, float h)
{
	this->w = w;
	this->h = h;
}

void GlButton::setColor(float r, float g, float b)
{
	this->r = r;
	this->g = g;
	this->b = b;
}


void GlButton::press()
{
	this->pressedDown = !this->pressedDown;
}

void GlButton::render()
{
	glBegin(GL_POLYGON);
	glColor3f(this->r, this->g, this->b); 
/*	if(this->pressedDown)
		glColor3f(1.0f, 1.0f, 0.7f); 
	else
		glColor3f(0.8f, 1.0f, 0.5f); 
*/
	glVertex2f(x,	y);
	glVertex2f(x+w,	y);
	glVertex2f(x+w,	y+h);
	glVertex2f(x,	y+h);

	glEnd();

}