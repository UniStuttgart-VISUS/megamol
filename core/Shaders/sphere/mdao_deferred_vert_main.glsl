in vec4 inPosition;

uniform float inWidth;
uniform float inHeight;

void main()
{
	gl_PointSize = max(inWidth, inHeight);
	gl_Position = inPosition;
}