
#include "LearningAgent.hpp"
#include "../util/Common.hpp"
#include "../util/Math.hpp"
#include "../util/Timer.hpp"
#include "../python/PythonContext.hpp"
#include "../python/PythonUtil.hpp"
#include "../python/NetworkSpec.hpp"
#include "../python/TFLearner.hpp"
#include "Constants.hpp"
#include "TrainingSample.hpp"

#include <boost/thread/shared_mutex.hpp>
#include <cassert>
#include <random>

using namespace learning;

struct LearningAgent::LearningAgentImpl {
  mutable boost::shared_mutex rwMutex;

  float pRandom;
  float temperature;

  python::PythonThreadContext ptctx;
  uptr<python::TFLearner> learner;

  unsigned itersSinceTargetUpdated = 0;

  LearningAgentImpl() : pRandom(0.0f), temperature(0.0001f), ptctx(python::GlobalContext()) {
    python::NetworkSpec spec(BOARD_WIDTH * BOARD_HEIGHT * 2,
                             GameAction::ALL_ACTIONS().size(),
                             MOMENTS_BATCH_SIZE);

    learner = make_unique<python::TFLearner>(ptctx, spec);
    itersSinceTargetUpdated = 0;
  }

  GameAction SelectAction(const GameState *state) {
    assert(state != nullptr);

    boost::shared_lock<boost::shared_mutex> lock(rwMutex);
    return chooseBestAction(*state, LearningAgent::EncodeGameState(state));
  }

  void SetPRandom(float pRandom) {
    assert(pRandom >= 0.0f && pRandom <= 1.0f);
    this->pRandom = pRandom;
  }

  void SetTemperature(float temperature) {
    assert(temperature > 0.0f);
    this->temperature = temperature;
  }

  GameAction SelectLearningAction(const GameState *state,
                                  const EVector &encodedState) {
    assert(state != nullptr);

    boost::shared_lock<boost::shared_mutex> lock(rwMutex);
    if (util::RandInterval(0.0, 1.0) < pRandom) {
      return chooseExplorativeAction(*state);
    } else {
      // return chooseWeightedAction(*state, encodedState);
      return chooseBestAction(*state, encodedState);
    }
  }

  void Learn(const vector<ExperienceMoment> &moments, float learnRate) {
    if (itersSinceTargetUpdated > TARGET_FUNCTION_UPDATE_RATE) {
      boost::unique_lock<boost::shared_mutex> lock(rwMutex);
      learner->UpdateTargetParams(ptctx);
      itersSinceTargetUpdated = 0;
    }
    itersSinceTargetUpdated++;

    vector<TrainingSample> learnSamples;
    learnSamples.reserve(moments.size());

    for (const auto &moment : moments) {
      learnSamples.emplace_back(moment.initialState, moment.successorState,
                                GameAction::ACTION_INDEX(moment.actionTaken),
                                moment.isSuccessorTerminal, moment.reward,
                                REWARD_DELAY_DISCOUNT);
    }

    // learningNet->Update(neuralnetwork::SamplesProvider(learnSamples),
    // learnRate);
  }

  void Finalise(void) {
    // obtain a write lock
    boost::unique_lock<boost::shared_mutex> lock(rwMutex);

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
    EMatrix qvalues = python::ToEigen2D(learner->QFunction(ptctx, python::ToNumpy(encodedState)));
    assert(qvalues.rows() == static_cast<int>(GameAction::ALL_ACTIONS().size()));
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

void LearningAgent::Write(std::ostream &out) { /*impl->targetNet->Write(out);*/
}

GameAction LearningAgent::SelectAction(const GameState *state) {
  return impl->SelectAction(state);
}

void LearningAgent::SetPRandom(float pRandom) { impl->SetPRandom(pRandom); }
void LearningAgent::SetTemperature(float temperature) {
  impl->SetTemperature(temperature);
}

GameAction LearningAgent::SelectLearningAction(const GameState *state,
                                               const EVector &encodedState) {
  return impl->SelectLearningAction(state, encodedState);
}

void LearningAgent::Learn(const vector<ExperienceMoment> &moments,
                          float learnRate) {
  impl->Learn(moments, learnRate);
}

void LearningAgent::Finalise(void) { impl->Finalise(); }

float LearningAgent::GetQValue(const GameState &state,
                               const GameAction &action) {
  return impl->GetQValue(state, action);
}
