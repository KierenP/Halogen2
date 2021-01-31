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

enum
{
    INPUT_LAYER,
    HIDDEN_LAYER_1,
    HIDDEN_LAYER_2,
    HIDDEN_LAYER_3,
    OUTPUT_LAYER,
};

constexpr size_t ARCHITECTURE[] = { 768, 256, 32, 32, 1 };

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

//No activation function
template<typename T>
struct nop {
    void operator()(T&) {}
};

//ReLU activation function
template<typename T>
struct relu {
    void operator()(T& i) { i = std::max(T(0), i); }
};

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class ACTIVATION>
class IncrementalLayer
{
public:
    void Init(float*& data);

    void RecalculateIncremental(std::array<T_in, INPUT> inputs);
    void ApplyDelta(const deltaArray& update);
    void ApplyInverseDelta();

    const std::array<T_out, OUTPUT>& GetActivation();

private:
    std::array<std::array<T_in, OUTPUT>, INPUT> weights;
    std::array<T_out, OUTPUT> bias;

    std::vector<std::array<T_out, OUTPUT>> Zeta;
    std::array<T_out, OUTPUT> output;

    ACTIVATION activation;
};

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class ACTIVATION>
class Layer
{
public:
    void Init(float*& data);

    void FeedForward(const std::array<T_in, INPUT>& input);
    const std::array<T_out, OUTPUT>& GetOutput();

private:
    std::array<std::array<T_in, INPUT>, OUTPUT> weights;
    std::array<T_out, OUTPUT> bias;

    std::array<T_out, OUTPUT> Zeta;
    ACTIVATION activation;
};

class Network
{
public:
    void RecalculateIncremental(std::array<int16_t, ARCHITECTURE[INPUT_LAYER]> inputs);
    void ApplyDelta(const deltaArray& update);  //incrementally update the connections between input layer and first hidden layer
    void ApplyInverseDelta();                   //for un-make moves
    int16_t Eval() const;                   //when used with above, this just calculates starting from the alpha of first hidden layer and skips input -> hidden

    static void Init();

private:
    static IncrementalLayer<int16_t, int16_t, ARCHITECTURE[INPUT_LAYER],    ARCHITECTURE[HIDDEN_LAYER_1], relu<int16_t>> layer1;
    static Layer           <int16_t, int32_t, ARCHITECTURE[HIDDEN_LAYER_1], ARCHITECTURE[HIDDEN_LAYER_2], relu<int32_t>> layer2;
    static Layer           <int32_t, int32_t, ARCHITECTURE[HIDDEN_LAYER_2], ARCHITECTURE[HIDDEN_LAYER_3], relu<int32_t>> layer3;
    static Layer           <int32_t, int32_t, ARCHITECTURE[HIDDEN_LAYER_3], ARCHITECTURE[OUTPUT_LAYER],    nop<int32_t>> layer4;
};
