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

class Network
{
public:
    void RecalculateIncremental(std::array<int16_t, INPUT_NEURONS> inputs);
    void ApplyDelta(const deltaArray& update);  //incrementally update the connections between input layer and first hidden layer
    void ApplyInverseDelta();                   //for un-make moves
    int16_t QuickEval() const;                  //when used with above, this just calculates starting from the alpha of first hidden layer and skips input -> hidden

    static void Init();

private:
    std::vector<std::array<int16_t, HIDDEN_NEURONS>> Zeta;

    static std::array<std::array<int16_t, HIDDEN_NEURONS>, INPUT_NEURONS> hiddenWeights;
    static std::array<int16_t, HIDDEN_NEURONS> hiddenBias;
    static std::array<int16_t, HIDDEN_NEURONS> outputWeights;
    static int16_t outputBias;
};

struct InputVector
{
    std::array<int16_t, 18> data = {};  //16 pawns and 2 kings
    int8_t size = 0;
};

class KP_Network
{
public:
    int16_t Eval(const InputVector& inputs) const;
    static void Init();

private:
    constexpr static size_t INPUT_NEURONS = 256;
    constexpr static size_t HIDDEN_NEURONS = 32;

    static void Aggregate(std::array<int16_t, HIDDEN_NEURONS>& zeta, const std::array<int16_t, HIDDEN_NEURONS>& weights);

    mutable std::array<int16_t, HIDDEN_NEURONS> zeta = {};

    static std::array<std::array<int16_t, HIDDEN_NEURONS>, INPUT_NEURONS> hiddenWeights;
    static std::array<int16_t, HIDDEN_NEURONS> hiddenBias;
    static std::array<int16_t, HIDDEN_NEURONS> outputWeights;
    static int16_t outputBias;
};

