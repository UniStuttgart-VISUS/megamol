#ifndef MMPROTEINPLUGIN_GLBUTTON_H_INCLUDED
#define MMPROTEINPLUGIN_GLBUTTON_H_INCLUDED

#include "GlWidget.h"
#include "GlWidgetLibrary.h"

//class GlWidgetLibrary;

class GlButton : public GlWidget
{
	friend class GlWidgetLibrary;

public:
	void move(float x, float y);
	void resize(float w, float h);
	void setColor(float r, float g, float b);
	void press();

private:
	GlButton(void);
	~GlButton(void);

	void render();

	float x, y, w, h;
	float r, g, b;
	volatile bool pressedDown;
};

#endif