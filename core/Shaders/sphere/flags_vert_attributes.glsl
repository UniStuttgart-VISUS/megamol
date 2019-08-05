#define FLAG_STORAGE

#ifdef FLAG_STORAGE
const uint FLAG_ENABLED = 1u << 0;
const uint FLAG_FILTERED = 1u << 1;
const uint FLAG_SELECTED = 1u << 2;
const uint FLAG_SOFTSELECTED = 1u << 3;

layout(std430, binding = 0) buffer Flags
{
	coherent uint flags[];
};

uniform uint flags_offset;

#endif // FLAG_STORAGE