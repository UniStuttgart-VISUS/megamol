#pragma once

/// <summary>
/// Sort all given arrays, according to the first in ascending order
/// </summary>
/// <template name="first_array_t">Type of the array which needs to be sorted</template>
/// <template name="array_ts">Types of the other arrays, that have to be reordered accordingly</template>
/// <param name="first_array">Array to sort</param>
/// <param name="arrays">Arrays to reorder accordingly</param>
template <typename first_array_t, typename... array_ts>
void sort(first_array_t& first_array, array_ts&... arrays);

/// <summary>
/// Sort all given arrays, according to the first one using the user-defined predicate
/// </summary>
/// <template name="predicate_t">Type of the predicate used for sorting</template>
/// <template name="array_ts">Types of the given arrays</template>
/// <param name="predicate">Predicate used for sorting</param>
/// <param name="arrays">Arrays to sort</param>
template <typename predicate_t, typename... array_ts>
void sort_with(predicate_t predicate, array_ts&... arrays);

#include "simultaneous_sort.inl"