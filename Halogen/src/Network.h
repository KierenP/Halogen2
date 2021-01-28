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
constexpr size_t HIDDEN_NEURONS = 256;

constexpr int16_t MAX_VALUE = 128;
constexpr int16_t PRECISION = ((size_t)std::numeric_limits<int16_t>::max() + 1) / MAX_VALUE;
constexpr int32_t SQUARE_PRECISION = (int32_t)PRECISION * PRECISION;
constexpr int32_t HALF_SQUARE_PRECISION = SQUARE_PRECISION / 2;
constexpr int16_t HALF_PRECISION = PRECISION / 2;

enum class VectorMode
{
    DENSE,
    SPARCE
};

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


struct InputVector
{
    std::array<int16_t, 32> data = {};
    int8_t size = 0;
};

class Network
{
public:
    void RecalculateIncremental(std::array<int16_t, INPUT_NEURONS> inputs);
    void ApplyDelta(const deltaArray& update);  //incrementally update the connections between input layer and first hidden layer
    void ApplyBlankDelta();
    void ApplyInverseDelta();                   //for un-make moves
    int16_t QuickEval() const;                  //when used with above, this just calculates starting from the alpha of first hidden layer and skips input -> hidden

    int16_t SparceEval(const InputVector& inputs) const;

    static void Init();

private:
    mutable std::vector<std::array<int16_t, HIDDEN_NEURONS>> zeta;

    static std::array<std::array<int16_t, HIDDEN_NEURONS>, INPUT_NEURONS> hiddenWeights;
    static std::array<int16_t, HIDDEN_NEURONS> hiddenBias;
    static std::array<int16_t, HIDDEN_NEURONS> outputWeights;
    static int16_t outputBias;
};

