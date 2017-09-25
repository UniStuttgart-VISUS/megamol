#include "stdafx.h"
#include "GlWidgetLibrary.h"
#include "GlWidget.h"
#include "GlButton.h"
#include "vislib/types.h"
#include "vislib/Array.h"
#include "mmcore/view/MouseFlags.h"

GlWidgetLibrary::GlWidgetLibrary(void)
: widgets(new vislib::Array<GlButton*>())
{
}

GlWidgetLibrary::~GlWidgetLibrary(void)
{
	/* delete all created widgets */
	
	/* delete widget container */
	delete widgets;
}

GlButton* GlWidgetLibrary::createButton()
{
	GlButton *button = new GlButton();
	this->widgets->Add(button);

	return button;
}

void GlWidgetLibrary::renderWidgets()
{
	SIZE_T elementCount = this->widgets->Count();
	for(SIZE_T i = 0; i < elementCount; ++i)
	{
		(*this->widgets)[i]->render();
	}
}

void GlWidgetLibrary::clear()
{
	SIZE_T elementCount = this->widgets->Count();
	for(SIZE_T i = 0; i < elementCount; ++i)
	{
		delete (*this->widgets)[i];
		this->widgets->Erase(i);
	}
}

void GlWidgetLibrary::mouseHandler(float x, float y, megamol::core::view::MouseFlags flags)
{
	using namespace megamol::core::view;

	if(flags & MOUSEFLAG_BUTTON_LEFT_DOWN && 
	   flags & MOUSEFLAG_BUTTON_LEFT_CHANGED)
	{
		SIZE_T elementCount = this->widgets->Count();
		for(SIZE_T i = 0; i < elementCount; ++i)
		{
			GlButton* widget = (*this->widgets)[i];
			if(x >= widget->x && x <= widget->x + widget->w &&
			   y >= widget->y && y <= widget->y + widget->h)
			{
				widget->press();
			}
		}
	}
}