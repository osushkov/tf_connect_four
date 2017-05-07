
#include "LearningAgent.hpp"
#include "../python/NetworkSpec.hpp"
#include "../python/PythonContext.hpp"
#include "../python/PythonUtil.hpp"
#include "../python/TFLearner.hpp"
#include "../util/Common.hpp"
#include "../util/Math.hpp"
#include "../util/Timer.hpp"
#include "Constants.hpp"
#include "TrainingSample.hpp"

#include <cassert>

using namespace learning;

struct LearningAgent::LearningAgentImpl {
  float pRandom;
  float temperature;

  python::PythonThreadContext ptctx;
  uptr<python::TFLearner> learner;

  unsigned itersSinceTargetUpdated;

  LearningAgentImpl()
      : pRandom(0.0f), temperature(0.0001f), ptctx(python::GlobalContext()) {
    python::PythonContextLock pl(ptctx);

    python::NetworkSpec spec(BOARD_WIDTH * BOARD_HEIGHT * 2,
                             GameAction::ALL_ACTIONS().size(),
                             MOMENTS_BATCH_SIZE);

    learner = make_unique<python::TFLearner>(spec);
    itersSinceTargetUpdated = 0;
  }

  GameAction SelectAction(const GameState *state) {
    assert(state != nullptr);
    return chooseBestAction(*state, LearningAgent::EncodeGameState(state));
  }

  GameAction SelectLearningAction(const GameState *state,
                                  const EVector &encodedState) {
    assert(state != nullptr);

    if (util::RandInterval(0.0, 1.0) < pRandom) {
      return chooseExplorativeAction(*state);
    } else {
      // return chooseWeightedAction(*state, encodedState);
      return chooseBestAction(*state, encodedState);
    }
  }

  void Learn(const vector<ExperienceMoment> &moments, float learnRate) {
    if (itersSinceTargetUpdated > TARGET_FUNCTION_UPDATE_RATE) {
      learner->UpdateTargetParams();
      itersSinceTargetUpdated = 0;
    }
    learner->Learn(makeQBatch(moments, learnRate));
    itersSinceTargetUpdated++;
  }

  void Finalise(void) {
    // if (learningNet == nullptr) {
    //   return;
    // }
    //
    // targetNet = learningNet->RefreshAndGetTarget();
    // learningNet.release();
    // learningNet = nullptr;
  }

  // TODO: probably dont need this.
  float GetQValue(const GameState &state, const GameAction &action) {
    EMatrix qvalues = learnerInference(LearningAgent::EncodeGameState(&state));
    return qvalues(GameAction::ACTION_INDEX(action), 0);
  }

  python::QLearnBatch makeQBatch(const vector<ExperienceMoment> &moments,
                                 float learnRate) {
    EMatrix initialStates(moments.size(), BOARD_WIDTH * BOARD_HEIGHT * 2);
    EMatrix successorStates(moments.size(), BOARD_WIDTH * BOARD_HEIGHT * 2);
    std::vector<int> actionsTaken(moments.size());
    std::vector<bool> isEndStateTerminal(moments.size());
    std::vector<float> rewardsGained(moments.size());

    for (unsigned i = 0; i < moments.size(); i++) {
      initialStates.row(i) = moments[i].initialState;
      successorStates.row(i) = moments[i].successorState;
      actionsTaken[i] = GameAction::ACTION_INDEX(moments[i].actionTaken);
      isEndStateTerminal[i] = moments[i].isSuccessorTerminal;
      rewardsGained[i] = moments[i].reward;
    }

    return python::QLearnBatch(
        python::ToNumpy(initialStates), python::ToNumpy(successorStates),
        python::ToNumpy(actionsTaken), python::ToNumpy(isEndStateTerminal),
        python::ToNumpy(rewardsGained), REWARD_DELAY_DISCOUNT, learnRate);
  }

  GameAction chooseBestAction(const GameState &state,
                              const EVector &encodedState) {
    EMatrix qvalues = learnerInference(encodedState);
    std::vector<unsigned> availableActions = state.AvailableActions();
    assert(availableActions.size() > 0);

    GameAction bestAction = GameAction::ACTION(availableActions[0]);
    float bestQValue = qvalues(availableActions[0], 0);

    for (unsigned i = 1; i < availableActions.size(); i++) {
      if (qvalues(availableActions[i]) > bestQValue) {
        bestQValue = qvalues(availableActions[i], 0);
        bestAction = GameAction::ACTION(availableActions[i]);
      }
    }
    return bestAction;
  }

  GameAction chooseExplorativeAction(const GameState &state) {
    auto aa = state.AvailableActions();
    return GameAction::ACTION(aa[rand() % aa.size()]);
  }

  GameAction chooseWeightedAction(const GameState &state,
                                  const EVector &encodedState) {
    EMatrix qvalues = learnerInference(encodedState);

    std::vector<unsigned> availableActions = state.AvailableActions();
    std::vector<float> weights;

    for (unsigned i = 0; i < availableActions.size(); i++) {
      weights.push_back(qvalues(availableActions[i], 0) / temperature);
    }
    weights = util::SoftmaxWeights(weights);

    float sample = util::RandInterval(0.0, 1.0);
    for (unsigned i = 0; i < weights.size(); i++) {
      sample -= weights[i];
      if (sample <= 0.0f) {
        // std::cout << "weighted: " << availableActions[i] << std::endl;
        return GameAction::ACTION(availableActions[i]);
      }
    }

    return chooseExplorativeAction(state);
  }

  EMatrix learnerInference(const EVector &encodedState) {
    EMatrix qvalues =
        python::ToEigen2D(learner->QFunction(python::ToNumpy(encodedState)))
            .transpose();
    std::cout << "wooo: " << qvalues << std::endl;
    std::cout << "bleh: " << qvalues.rows() << " " << qvalues.cols()
              << std::endl;
    assert(qvalues.rows() ==
           static_cast<int>(GameAction::ALL_ACTIONS().size()));
    assert(qvalues.cols() == 1);

    return qvalues;
  }
};

EVector LearningAgent::EncodeGameState(const GameState *state) {
  EVector result(2 * BOARD_WIDTH * BOARD_HEIGHT);
  result.fill(0.0f);

  for (unsigned r = 0; r < BOARD_HEIGHT; r++) {
    for (unsigned c = 0; c < BOARD_WIDTH; c++) {
      unsigned ri = 2 * (c + r * BOARD_WIDTH);

      switch (state->GetCell(r, c)) {
      case CellState::MY_TOKEN:
        result(ri) = 1.0f;
        break;
      case CellState::OPPONENT_TOKEN:
        result(ri + 1) = 1.0f;
        break;
      default:
        break;
      }

      ri++;
    }
  }

  return result;
}

LearningAgent::LearningAgent() : impl(new LearningAgentImpl()) {}
LearningAgent::~LearningAgent() = default;

uptr<LearningAgent> LearningAgent::Read(std::istream &in) {
  uptr<LearningAgent> result = make_unique<LearningAgent>();
  // result->impl->targetNet = neuralnetwork::Network::Read(in);
  // result->impl->learningNet.release();
  // result->impl->learningNet = nullptr;
  return result;
}

void LearningAgent::Write(std::ostream &out) {
  python::PythonContextLock pl(impl->ptctx);
  /*impl->targetNet->Write(out);*/
}

GameAction LearningAgent::SelectAction(const GameState *state) {
  python::PythonContextLock pl(impl->ptctx);
  return impl->SelectAction(state);
}

void LearningAgent::SetPRandom(float pRandom) { impl->pRandom = pRandom; }

void LearningAgent::SetTemperature(float temperature) {
  impl->temperature = temperature;
}

GameAction LearningAgent::SelectLearningAction(const GameState *state,
                                               const EVector &encodedState) {
  python::PythonContextLock pl(impl->ptctx);
  return impl->SelectLearningAction(state, encodedState);
}

void LearningAgent::Learn(const vector<ExperienceMoment> &moments,
                          float learnRate) {
  python::PythonContextLock pl(impl->ptctx);
  impl->Learn(moments, learnRate);
}

void LearningAgent::Finalise(void) {
  python::PythonContextLock pl(impl->ptctx);
  impl->Finalise();
}

float LearningAgent::GetQValue(const GameState &state,
                               const GameAction &action) {
  python::PythonContextLock pl(impl->ptctx);
  return impl->GetQValue(state, action);
}
