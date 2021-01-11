#include "Network.h"
#include "7a4edf1c5c72c9e3415ac01d2bf534170cdaa1d82e99876a0bf35ef5800b7fac.nn"

std::array<std::array<int16_t, HIDDEN_NEURONS>, INPUT_NEURONS> Network::hiddenWeights = {};
std::array<int16_t, HIDDEN_NEURONS> Network::hiddenBias = {};
std::array<int16_t, HIDDEN_NEURONS> Network::outputWeights = {};
int16_t Network::outputBias = {};

void Network::Init()
{
    size_t index = 0;

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        hiddenBias[i] = (int16_t)round(label[index++] * PRECISION);

    for (size_t i = 0; i < INPUT_NEURONS; i++)
        for (size_t j = 0; j < HIDDEN_NEURONS; j++)
            hiddenWeights[i][j] = (int16_t)round(label[index++] * PRECISION);

    outputBias = (int16_t)round(label[index++] * PRECISION);

    for (size_t i = 0; i < HIDDEN_NEURONS; i++)
        outputWeights[i] = (int16_t)round(label[index++] * PRECISION);
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