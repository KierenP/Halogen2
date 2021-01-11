#include "Network.h"
#include "b9e24f227925572461787cdafb807046a6e0f29a2283017c227508d2280c0fea.nn"

std::array<std::array<int16_t, HIDDEN_NEURONS>, INPUT_NEURONS> Network::hiddenWeights = {};
std::array<int16_t, HIDDEN_NEURONS> Network::hiddenBias = {};
std::array<int16_t, HIDDEN_NEURONS> Network::outputWeights = {};
int16_t Network::outputBias = {};

void Network::Init()
{
    size_t index = 0;

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        hiddenBias[i] = label[index++];

    for (size_t i = 0; i < INPUT_NEURONS; i++)
        for (size_t j = 0; j < HIDDEN_NEURONS; j++)
            hiddenWeights[i][j] = label[index++];

    outputBias = label[index++];

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        outputWeights[i] = label[index++];
}

void Network::RecalculateIncremental(std::array<int16_t, INPUT_NEURONS> inputs)
{
    Zeta.resize(1);

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        Zeta[0][i] = hiddenBias[i];

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        for (size_t j = 0; j < INPUT_NEURONS; j++)
            Zeta[0][i] += inputs[j] * hiddenWeights[j][i];
}

void Network::ApplyDelta(const deltaArray& update)
{
    Zeta.push_back(Zeta.back());

    for (size_t i = 0; i < update.size; i++)
    {
        if (update.deltas[i].delta == 1)
            for (size_t j = 0; j < HIDDEN_NEURONS; j++)
                Zeta.back()[j] += hiddenWeights[update.deltas[i].index][j];
        else
            for (size_t j = 0; j < HIDDEN_NEURONS; j++)
                Zeta.back()[j] -= hiddenWeights[update.deltas[i].index][j];
    }
}

void Network::ApplyInverseDelta()
{
    Zeta.pop_back();
}

int16_t Network::QuickEval() const
{
    int32_t output = outputBias * PRECISION;

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        output += std::max(int16_t(0), Zeta.back()[i]) * outputWeights[i];

    return output / SQUARE_PRECISION;
}