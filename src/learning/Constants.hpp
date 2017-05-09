
#pragma once

namespace learning {

static constexpr unsigned MOMENTS_BATCH_SIZE = 256;
static constexpr unsigned TARGET_FUNCTION_UPDATE_RATE = 5000;
static constexpr float REWARD_DELAY_DISCOUNT = 1.0f;

static constexpr unsigned EXPERIENCE_MEMORY_SIZE = 100000;

static constexpr float INITIAL_PRANDOM = 0.3f;
static constexpr float TARGET_PRANDOM = 0.01f;

static constexpr float INITIAL_LEARN_RATE = 0.01f;
static constexpr float TARGET_LEARN_RATE = 0.0009f;
}
