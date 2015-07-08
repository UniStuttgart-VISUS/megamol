#ifndef MMPROTEINPLUGIN_GLWIDGETLIBRARY_H_INCLUDED
#define MMPROTEINPLUGIN_GLWIDGETLIBRARY_H_INCLUDED

#include "vislib/Array.h"
#include "mmcore/view/MouseFlags.h"

class GlWidget;
class GlButton;

namespace vislib {

//template <class T> class Array;
}

class GlWidgetLibrary
{
public:
	GlWidgetLibrary(void);
	~GlWidgetLibrary(void);

	GlButton* createButton();
	void renderWidgets();
	void clear();
	void mouseHandler(float x, float y, megamol::core::view::MouseFlags flags);
private:
	vislib::Array<GlButton*> *widgets;
};

#endif