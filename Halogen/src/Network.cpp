#include "Network.h"
#include "halogen-x256-eb873cf4.nn"

IncrementalLayer<INPUT_TYPE,  HIDDEN_TYPE, ARCHITECTURE[INPUT_LAYER],    ARCHITECTURE[HIDDEN_LAYER_1], relu<HIDDEN_TYPE>> Network::layer1;
Layer           <HIDDEN_TYPE, OUTPUT_TYPE, ARCHITECTURE[HIDDEN_LAYER_1], ARCHITECTURE[OUTPUT_LAYER],   nop <OUTPUT_TYPE>> Network::layer2;

void Network::Init()
{
    auto Data = reinterpret_cast<float*>(label);
    layer1.Init(Data);
    layer2.Init(Data);
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class activation>
void Layer<T_in, T_out, INPUT, OUTPUT, activation>::Init(float*& data)
{
    for (size_t i = 0; i < OUTPUT; i++)
        bias[i] = (T_in)round(*data++ * PRECISION);

    for (size_t i = 0; i < INPUT; i++)
        for (size_t j = 0; j < OUTPUT; j++)
            weights[j][i] = (T_in)round(*data++ * PRECISION);
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class activation>
inline void IncrementalLayer<T_in, T_out, INPUT, OUTPUT, activation>::Init(float*& data)
{
    for (size_t i = 0; i < OUTPUT; i++)
        bias[i] = (T_in)round(*data++ * PRECISION);

    for (size_t i = 0; i < INPUT; i++)
        for (size_t j = 0; j < OUTPUT; j++)
            weights[i][j] = (T_in)round(*data++ * PRECISION);
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class activation>
void IncrementalLayer<T_in, T_out, INPUT, OUTPUT, activation>::RecalculateIncremental(std::array<T_in, INPUT> inputs)
{
    Zeta.resize(1);

    for (size_t i = 0; i < OUTPUT; i++)
        Zeta[0][i] = bias[i];

    for (size_t i = 0; i < OUTPUT; i++)
        for (size_t j = 0; j < INPUT; j++)
            Zeta[0][i] += inputs[j] * weights[j][i];
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class activation>
void IncrementalLayer<T_in, T_out, INPUT, OUTPUT, activation>::ApplyDelta(const deltaArray& update)
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

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class activation>
void IncrementalLayer<T_in, T_out, INPUT, OUTPUT, activation>::ApplyInverseDelta()
{
    Zeta.pop_back();
}

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class activation>
const std::array<T_out, OUTPUT>& IncrementalLayer<T_in, T_out, INPUT, OUTPUT, activation>::GetActivation()
{
    activation function;

    for (size_t i = 0; i < OUTPUT; i++)
    {   
        output[i] = static_cast<T_out>(Zeta.back()[i]);
        function(output[i]);
    }

    return output;
}

template <typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, class activation>
const std::array<T_out, OUTPUT>& Layer<T_in, T_out, INPUT, OUTPUT, activation>::GetActivation()
{
    activation function;

    for (size_t i = 0; i < OUTPUT; i++)
    {
        output[i] = static_cast<T_out>(Zeta[i]);
        function(output[i]);
    }

    return output;
}

template<typename T_in, typename T_out, size_t INPUT, size_t OUTPUT, typename activation>
void Layer<T_in, T_out, INPUT, OUTPUT, activation>::FeedForward(const std::array<T_in, INPUT>& input)
{
    for (size_t i = 0; i < OUTPUT; i++)
        Zeta[i] = static_cast<T_out>(bias[i]);

    for (size_t i = 0; i < OUTPUT; i++)
        for (size_t j = 0; j < INPUT; j++)
            Zeta[i] += input[j] * weights[i][j];
}

void Network::RecalculateIncremental(std::array<INPUT_TYPE, ARCHITECTURE[INPUT_LAYER]> inputs)
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

OUTPUT_TYPE Network::Eval() const
{
    layer2.FeedForward(layer1.GetActivation());
    return layer2.GetActivation()[0] / SQUARE_PRECISION;
}