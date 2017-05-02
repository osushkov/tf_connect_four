
#include "Trainer.hpp"
#include "util/Common.hpp"
#include "util/Timer.hpp"
#include "connectfour/GameAction.hpp"
#include "connectfour/GameRules.hpp"
#include "connectfour/GameState.hpp"
#include "learning/Constants.hpp"
#include "learning/ExperienceMemory.hpp"
#include "learning/LearningAgent.hpp"
#include "learning/RandomAgent.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <future>
#include <thread>
#include <vector>

using namespace learning;

static constexpr unsigned EXPERIENCE_MEMORY_SIZE = 1000000;

static constexpr float INITIAL_PRANDOM = 0.5f;
static constexpr float TARGET_PRANDOM = 0.01f;

static constexpr float INITIAL_TEMPERATURE = 0.5f;
static constexpr float TARGET_TEMPERATURE = 0.001f;

static constexpr float INITIAL_LEARN_RATE = 0.1f;
static constexpr float TARGET_LEARN_RATE = 0.001f;


struct PlayoutAgent {
  LearningAgent *agent;
  ExperienceMemory *memory;

  vector<EVector> stateHistory;
  vector<GameAction> actionHistory;

  PlayoutAgent(LearningAgent *agent, ExperienceMemory *memory) : agent(agent), memory(memory) {}

  bool havePreviousState(void) { return stateHistory.size() > 0; }

  void addTransitionToMemory(EVector &curState, float reward, bool isTerminal) {
    if (!havePreviousState()) {
      return;
    }

    EVector &prevState = stateHistory[stateHistory.size() - 1];
    GameAction &performedAction = actionHistory[actionHistory.size() - 1];

    memory->AddExperience(
        ExperienceMoment(prevState, performedAction, curState, reward, isTerminal));
  }

  void addMoveToHistory(const EVector &state, const GameAction &action) {
    stateHistory.push_back(state);
    actionHistory.push_back(action);
  }
};

struct Trainer::TrainerImpl {
  vector<ProgressCallback> callbacks;
  atomic<unsigned> numLearnIters;

  void AddProgressCallback(ProgressCallback callback) { callbacks.push_back(callback); }

  uptr<LearningAgent> TrainAgent(unsigned iters) {
    auto experienceMemory =
        make_unique<ExperienceMemory>(EXPERIENCE_MEMORY_SIZE);

    uptr<LearningAgent> agent = make_unique<LearningAgent>();
    trainAgent(agent.get(), experienceMemory.get(), iters, INITIAL_PRANDOM,
               TARGET_PRANDOM);

    return move(agent);
  }

  void trainAgent(LearningAgent *agent, ExperienceMemory *memory,
                  unsigned iters, float initialPRandom, float targetPRandom) {

    numLearnIters = 0;

    std::thread playoutThread =
        startPlayoutThread(agent, memory, iters, initialPRandom, targetPRandom);

    std::thread learnThread = startLearnThread(agent, memory, iters);

    playoutThread.join();
    learnThread.join();
  }

  std::thread startPlayoutThread(LearningAgent *agent, ExperienceMemory *memory,
                                 unsigned iters, float initialPRandom,
                                 float targetPRandom) {

    return std::thread([this, agent, memory, iters, initialPRandom, targetPRandom]() {
      float pRandDecay = powf(targetPRandom / initialPRandom, 1.0f / iters);
      assert(pRandDecay > 0.0f && pRandDecay <= 1.0f);

      float tempDecay = powf(TARGET_TEMPERATURE / INITIAL_TEMPERATURE, 1.0f / iters);
      assert(tempDecay > 0.0f && tempDecay <= 1.0f);

      while (true) {
        unsigned doneIters = numLearnIters.load();
        if (doneIters >= iters) {
          break;
        }

        float prand = initialPRandom * powf(pRandDecay, doneIters);
        float temp = INITIAL_TEMPERATURE * powf(tempDecay, doneIters);

        agent->SetPRandom(prand);
        agent->SetTemperature(temp);

        this->playoutRoundVsSelf(agent, memory);
      }
    });
  }

  std::thread startLearnThread(LearningAgent *agent, ExperienceMemory *memory, unsigned iters) {
    return std::thread([this, agent, memory, iters]() {
      while (memory->NumMemories() < 10 * MOMENTS_BATCH_SIZE) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      float learnRateDecay = powf(TARGET_LEARN_RATE / INITIAL_LEARN_RATE, 1.0f / iters);
      assert(learnRateDecay > 0.0f && learnRateDecay < 1.0f);

      for (unsigned i = 0; i < iters; i++) {
        for (auto &cb : this->callbacks) {
          cb(agent, i);
        }

        float learnRate = INITIAL_LEARN_RATE * powf(learnRateDecay, i);
        agent->Learn(memory->Sample(MOMENTS_BATCH_SIZE), 1.0f);
        this->numLearnIters++;
      }

      agent->Finalise();
    });
  }

  void playoutRoundVsSelf(LearningAgent *agent, ExperienceMemory *memory) {
    GameRules *rules = GameRules::Instance();
    GameState curState = generateStartState();

    std::vector<PlayoutAgent> playoutAgents = {PlayoutAgent(agent, memory),
                                               PlayoutAgent(agent, memory)};
    unsigned curPlayerIndex = 0;
    while (true) {
      PlayoutAgent &curPlayer = playoutAgents[curPlayerIndex];
      PlayoutAgent &otherPlayer = playoutAgents[(curPlayerIndex + 1) % 2];

      EVector encodedState = LearningAgent::EncodeGameState(&curState);
      GameAction action = curPlayer.agent->SelectLearningAction(&curState, encodedState);

      curPlayer.addTransitionToMemory(encodedState, 0.0f, false);
      curPlayer.addMoveToHistory(encodedState, action);

      curState = curState.SuccessorState(action);

      switch (rules->GameCompletionState(curState)) {
      case CompletionState::WIN:
        encodedState = LearningAgent::EncodeGameState(&curState);
        curPlayer.addTransitionToMemory(encodedState, 1.0f, true);
        otherPlayer.addTransitionToMemory(encodedState, -1.0f, true);
        return;
      case CompletionState::LOSS:
        assert(false); // This actually shouldn't be possible.
        return;
      case CompletionState::DRAW:
        encodedState = LearningAgent::EncodeGameState(&curState);
        curPlayer.addTransitionToMemory(encodedState, 0.0f, true);
        otherPlayer.addTransitionToMemory(encodedState, 0.0f, true);
        return;
      case CompletionState::UNFINISHED:
        curState.FlipState();
        curPlayerIndex = (curPlayerIndex + 1) % 2;
        break;
      }
    }
  }

  GameState generateStartState(void) {
    GameRules *rules = GameRules::Instance();

    RandomAgent agent;
    std::vector<GameState> states;

    GameState curState(rules->InitialState());
    bool isFinished = false;

    while (!isFinished) {
      states.push_back(curState);
      GameAction action = agent.SelectAction(&curState);
      curState = curState.SuccessorState(action);

      switch (rules->GameCompletionState(curState)) {
      case CompletionState::WIN:
      case CompletionState::LOSS:
      case CompletionState::DRAW:
        isFinished = true;
        break;
      case CompletionState::UNFINISHED:
        curState.FlipState();
        break;
      }
    }

    unsigned backtrack = 4;
    if (states.size() <= backtrack) {
      return states[0];
    } else {
      return states[rand() % (states.size() - backtrack)];
    }
  }
};

Trainer::Trainer() : impl(new TrainerImpl()) {}
Trainer::~Trainer() = default;

void Trainer::AddProgressCallback(ProgressCallback callback) {
  impl->AddProgressCallback(callback);
}

uptr<LearningAgent> Trainer::TrainAgent(unsigned iters) { return impl->TrainAgent(iters); }
