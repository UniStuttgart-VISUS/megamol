void AddSample( float ssaoValue, float edgeValue, inout float sum, inout float sumWeight )
{
    float weight = edgeValue;

    sum += (weight * ssaoValue);
    sumWeight += weight;
}
