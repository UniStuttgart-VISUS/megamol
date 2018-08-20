#version 460

//in vec2 UV;
//uniform sampler2D sim_texture;

in vec3 UVW;
uniform sampler3D sim_texture;
uniform sampler2D alphaTex;

out vec4 out_color;

void main() { 
	vec4 value = texture(sim_texture, UVW);
	float val = value.x;

	// get energy function
	// i.e. get uv texture coordinate for alphaTex
	vec2 alphaUV;
	if(val >= 0.f && val < 1.5f)		// section around 1keV
	{
		alphaUV.y = 1.f / 16.f;
	}
	else if(val >= 1.5 && val < 3.5f)	// section around 2keV
	{
		alphaUV.y = 3.f / 16.f;
	}
	else if(val >= 3.5 && val < 7.5f)	// section around 5keV
	{
		alphaUV.y = 5.f / 16.f;
	}
	else if(val >= 7.5 && val < 12.5f)	// section around 10keV
	{
		alphaUV.y = 7.f / 16.f;
	}
	else if(val >= 12.5 && val < 17.5f)	// section around 15keV
	{
		alphaUV.y = 9.f / 16.f;
	}
	else if(val >= 17.5 && val < 22.5f)	// section around 20keV
	{
		alphaUV.y = 11.f / 16.f;
	}
	else if(val >= 22.5 && val < 27.5f)	// section around 25keV
	{
		alphaUV.y = 13.f / 16.f;
	}
	else if(val >= 27.5f)				// section above 30keV
	{
		alphaUV.y = 15.f / 16.f;
	}
	alphaUV.x = UVW.y;
	//alphaUV.y = 15.f / 16.f;
	float alpha = texture(alphaTex, alphaUV).x;

	int mode = 2;
	if(mode == 0)
	{
		if (val <= 0) discard;
		out_color = vec4(val, val, val, alpha);
	} 
	else if(mode == 1) 
	{
		if(val <= 0.f) 
		{
			discard;
			out_color = vec4(0.f, 0.f, -val, alpha);
		} 
		else if (val.r <= 0.4f)
		{
			out_color = vec4(0.f, val, 0.f, alpha);
		} 
		else 
		{
			out_color = vec4(0.1f, val, 0.05f, alpha);
		} 
	} 
	else if(mode == 2) 
	{
		//if(val * alpha <= 0.05f) discard;
		vec4 color = vec4(0.f, val * alpha * 2.f, 0.f, val * alpha);
		float brightness = length(color.xyz);
		if(val >= 0.1f) out_color = vec4(0.f, val * alpha * 2.f, 0.f, val * alpha);
	} 
}