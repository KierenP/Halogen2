#pragma once
#include <vector>
#include <memory>		//required to compile with g++
#include <array>
#include <assert.h>
#include <iostream>

constexpr size_t TableSize = 65536;

struct EvalCacheEntry
{
	uint64_t key = 0;
	int eval = -1;
};

class EvalCacheTable
{
public:
	void AddEntry(uint64_t key, int eval);
	bool GetEntry(uint64_t key, int& eval);

	void Reset();

private:
	std::vector<EvalCacheEntry> table = std::vector<EvalCacheEntry>(TableSize);
};
