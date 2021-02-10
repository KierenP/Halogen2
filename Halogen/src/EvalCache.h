#pragma once
#include <vector>
#include <memory>		//required to compile with g++
#include <array>
#include <assert.h>
#include <iostream>

struct EvalCacheEntry
{
	uint32_t key = 0;
	short int eval = 0;
};

class EvalCacheTable
{
public:
	EvalCacheTable() { Reset(); }
	~EvalCacheTable() = default;

	void AddEntry(uint64_t key, int eval);
	bool GetEntry(uint64_t key, int& eval);

	void Reset();

private:
	std::vector<EvalCacheEntry> table;
};
