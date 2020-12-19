#pragma once
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <assert.h>
#include <random>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cstring>
#include "EvalCache.h"
#include "BitBoardDefine.h"

constexpr size_t INPUT_NEURONS = 12 * 64;
constexpr size_t HIDDEN_NEURONS_1 = 256;
constexpr size_t HIDDEN_NEURONS_2 = 32;

constexpr int16_t MAX_VALUE = 128;
constexpr int16_t PRECISION = ((size_t)std::numeric_limits<int16_t>::max() + 1) / MAX_VALUE;
constexpr int32_t SQUARE_PRECISION = (int32_t)PRECISION * PRECISION;
constexpr int32_t HALF_SQUARE_PRECISION = SQUARE_PRECISION / 2;
constexpr int16_t HALF_PRECISION = PRECISION / 2;

struct deltaArray
{
    struct deltaPoint
    {
        size_t index;
        int16_t delta;
    };

    size_t size;
    deltaPoint deltas[4];
};

void RecalculateIncremental(std::array<int16_t, INPUT_NEURONS> inputs, std::array<std::array<int16_t, HIDDEN_NEURONS_1>, MAX_DEPTH + 1>& Zeta, size_t& incrementalDepth);
void ApplyDelta(deltaArray& update, std::array<std::array<int16_t, HIDDEN_NEURONS_1>, MAX_DEPTH + 1>& Zeta, size_t& incrementalDepth);     //incrementally update the connections between input layer and first hidden layer
void ApplyInverseDelta(size_t& incrementalDepth);                                                                                   //for un-make moves
int16_t QuickEval(const std::array<std::array<int16_t, HIDDEN_NEURONS_1>, MAX_DEPTH + 1>& Zeta, const size_t& incrementalDepth);          //when used with above, this just calculates starting from the alpha of first hidden layer and skips input -> hidden

void NetworkInit();