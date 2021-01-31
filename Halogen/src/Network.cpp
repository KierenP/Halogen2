#include "Network.h"
#include "epoch30.net"

IncrementalLayer<int16_t, int16_t, ARCHITECTURE[INPUT_LAYER],    ARCHITECTURE[HIDDEN_LAYER_1], relu<int16_t>> Network::layer1;
Layer           <int16_t, int32_t, ARCHITECTURE[HIDDEN_LAYER_1], ARCHITECTURE[HIDDEN_LAYER_2], relu<int32_t>> Network::layer2;
Layer           <int32_t, int32_t, ARCHITECTURE[HIDDEN_LAYER_2], ARCHITECTURE[HIDDEN_LAYER_3], relu<int32_t>> Network::layer3;
Layer           <int32_t, int32_t, ARCHITECTURE[HIDDEN_LAYER_3], ARCHITECTURE[OUTPUT_LAYER],    nop<int32_t>> Network::layer4;

void Network::Init()
{
    auto Data = reinterpret_cast<float*>(label);
    layer1.Init(Data);
    layer2.Init(Data);
    layer3.Init(Data);
    layer4.Init(Data);
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class ACTIVATION>
void Layer<T_in, T_out, INPUT, OUTPUT, ACTIVATION>::Init(float*& data)
{
    for (size_t i = 0; i < OUTPUT; i++)
        bias[i] = (T_in)round(*data++ * PRECISION);

    for (size_t i = 0; i < INPUT; i++)
        for (size_t j = 0; j < OUTPUT; j++)
            weights[j][i] = (T_in)round(*data++ * PRECISION);
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class ACTIVATION>
inline void IncrementalLayer<T_in, T_out, INPUT, OUTPUT, ACTIVATION>::Init(float*& data)
{
    for (size_t i = 0; i < OUTPUT; i++)
        bias[i] = (T_in)round(*data++ * PRECISION);

    for (size_t i = 0; i < INPUT; i++)
        for (size_t j = 0; j < OUTPUT; j++)
            weights[i][j] = (T_in)round(*data++ * PRECISION);
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class ACTIVATION>
void IncrementalLayer<T_in, T_out, INPUT, OUTPUT, ACTIVATION>::RecalculateIncremental(std::array<T_in, INPUT> inputs)
{
    Zeta = { bias };

    for (size_t i = 0; i < OUTPUT; i++)
        for (size_t j = 0; j < INPUT; j++)
            Zeta[0][i] += inputs[j] * weights[j][i];
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class ACTIVATION>
void IncrementalLayer<T_in, T_out, INPUT, OUTPUT, ACTIVATION>::ApplyDelta(const deltaArray& update)
{
    Zeta.push_back(Zeta.back());

    for (size_t i = 0; i < update.size; i++)
    {
        if (update.deltas[i].delta == 1)
            for (size_t j = 0; j < OUTPUT; j++)
                Zeta.back()[j] += weights[update.deltas[i].index][j];
        else
            for (size_t j = 0; j < OUTPUT; j++)
                Zeta.back()[j] -= weights[update.deltas[i].index][j];
    }
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class ACTIVATION>
void IncrementalLayer<T_in, T_out, INPUT, OUTPUT, ACTIVATION>::ApplyInverseDelta()
{
    Zeta.pop_back();
}

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class ACTIVATION>
const std::array<T_out, OUTPUT>& IncrementalLayer<T_in, T_out, INPUT, OUTPUT, ACTIVATION>::GetActivation()
{
    output = Zeta.back();

    for (size_t i = 0; i < OUTPUT; i++)
        activation(output[i]);

    return output;
}

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class ACTIVATION>
const std::array<T_out, OUTPUT>& Layer<T_in, T_out, INPUT, OUTPUT, ACTIVATION>::GetOutput()
{
    return Zeta;
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, typename ACTIVATION>
void Layer<T_in, T_out, INPUT, OUTPUT, ACTIVATION>::FeedForward(const std::array<T_in, INPUT>& input)
{
    Zeta = bias;

    for (size_t i = 0; i < OUTPUT; i++)
    {
        for (size_t j = 0; j < INPUT; j++)
            Zeta[i] += input[j] * weights[i][j] / PRECISION;

        activation(Zeta[i]);
    }    
}

void Network::RecalculateIncremental(std::array<int16_t, ARCHITECTURE[INPUT_LAYER]> inputs)
{
    layer1.RecalculateIncremental(inputs);
}

void Network::ApplyDelta(const deltaArray& update)
{
    layer1.ApplyDelta(update);
}

void Network::ApplyInverseDelta()
{
    layer1.ApplyInverseDelta();
}

int16_t Network::Eval() const
{
    layer2.FeedForward(layer1.GetActivation());
    layer3.FeedForward(layer2.GetOutput());
    layer4.FeedForward(layer3.GetOutput());
    return layer4.GetOutput()[0] / PRECISION;
}