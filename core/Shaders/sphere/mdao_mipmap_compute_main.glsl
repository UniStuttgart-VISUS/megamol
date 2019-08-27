
#version 430

layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

uniform readonly layout(binding = 0, r8) image3D inputImage;
uniform writeonly layout(binding = 1, r8) image3D outputImage;

void main()
{
	ivec3 gid = ivec3(gl_GlobalInvocationID);
	ivec3 inputSize = imageSize(inputImage) - 1;
	// Get the 8 texel from the last stage
	ivec3 pos = gid*2;
	
	vec4 result = vec4(0.0);
	ivec3 posMax = min(pos+1, inputSize);
	
	for (int x=pos.x; x<=posMax.x; ++x) 
		for (int y=pos.y; y<=posMax.y; ++y) 
			for (int z=pos.z; z<=posMax.z; ++z)
				result += imageLoad(inputImage, ivec3(x,y,z));

	result /= 8.0;
	
	imageStore(outputImage, gid, result);
}
