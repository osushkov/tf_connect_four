#pragma once

#include "../util/Math.hpp"

namespace learning {

struct TrainingSample {
  EVector startState;
  EVector endState;
  unsigned actionTaken;

  bool isEndStateTerminal;
  float rewardGained;
  float futureRewardDiscount;

  TrainingSample(const EVector &startState, const EVector &endState, unsigned actionTaken,
                 bool isEndStateTerminal, float rewardGained, float futureRewardDiscount)
      : startState(startState), endState(endState), actionTaken(actionTaken),
        isEndStateTerminal(isEndStateTerminal), rewardGained(rewardGained),
        futureRewardDiscount(futureRewardDiscount) {}
};
}
