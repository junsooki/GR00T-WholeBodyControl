#include <gtest/gtest.h>

#include "sim_lockstep.hpp"
#include "state_logger.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

namespace {

LockstepTokenBundle Token(uint32_t session, uint32_t step, uint32_t action) {
  LockstepTokenBundle bundle;
  bundle.envelope = {LOCKSTEP_MAGIC, session, step, action};
  bundle.frame_index = action;
  bundle.token.assign(64, static_cast<double>(action));
  return bundle;
}

uint64_t LogOne(StateLogger& logger) {
  std::array<double, 4> quat{};
  std::array<double, 3> xyz{};
  std::array<double, 29> body{};
  std::array<double, 58> temps{};
  std::array<double, 7> hand{};
  return logger.LogFullState(
      quat, xyz, xyz, quat, xyz, xyz, body, body, body, temps, body, body,
      hand, hand, hand, hand, hand, hand);
}

}  // namespace

TEST(SimLockstep, MatchesEitherArrivalOrderAndCommitsOnce) {
  SimLockstepGate gate;
  const LockstepEnvelope ready{LOCKSTEP_MAGIC, 7, 0, LOCKSTEP_NO_ACTION};
  EXPECT_EQ(gate.PushState(ready), SimLockstepGate::PushResult::Accepted);
  auto work = gate.Wait(std::stop_token{});
  ASSERT_TRUE(work.has_value());
  EXPECT_EQ(work->kind, SimLockstepGate::WorkKind::SessionReady);
  EXPECT_TRUE(gate.Commit(*work));

  const auto token = Token(7, 1, 0);
  EXPECT_EQ(gate.PushToken(token), SimLockstepGate::PushResult::Accepted);
  EXPECT_EQ(gate.PushState(token.envelope), SimLockstepGate::PushResult::Accepted);
  work = gate.Wait(std::stop_token{});
  ASSERT_TRUE(work.has_value());
  EXPECT_EQ(work->kind, SimLockstepGate::WorkKind::Control);
  EXPECT_TRUE(gate.Commit(*work));
  EXPECT_EQ(gate.LastStep(), 1u);
  ASSERT_TRUE(gate.LastAction().has_value());
  EXPECT_EQ(*gate.LastAction(), 0u);
  ASSERT_TRUE(gate.LoadCommitted());
  EXPECT_EQ(gate.LoadCommitted()->envelope, token.envelope);

  EXPECT_EQ(gate.PushState(token.envelope), SimLockstepGate::PushResult::Duplicate);
  EXPECT_EQ(gate.PushToken(token), SimLockstepGate::PushResult::Duplicate);
  EXPECT_EQ(gate.LastStep(), 1u);
}

TEST(SimLockstep, RejectsWrongSessionJumpsAndBadFrame) {
  SimLockstepGate gate;
  EXPECT_EQ(
      gate.PushState({LOCKSTEP_MAGIC, 9, 1, 0}),
      SimLockstepGate::PushResult::Rejected);
  EXPECT_EQ(
      gate.PushState({LOCKSTEP_MAGIC, 7, 0, LOCKSTEP_NO_ACTION}),
      SimLockstepGate::PushResult::Accepted);
  auto ready = gate.Wait(std::stop_token{});
  ASSERT_TRUE(ready.has_value());
  ASSERT_TRUE(gate.Commit(*ready));
  EXPECT_EQ(
      gate.PushState({LOCKSTEP_MAGIC, 7, 2, 0}),
      SimLockstepGate::PushResult::Rejected);
  auto bad = Token(7, 1, 0);
  bad.frame_index = 5;
  EXPECT_EQ(gate.PushToken(bad), SimLockstepGate::PushResult::Rejected);
}

TEST(SimLockstep, NoActionControlDoesNotAdvanceActionSequence) {
  SimLockstepGate gate;
  gate.PushState({LOCKSTEP_MAGIC, 4, 0, LOCKSTEP_NO_ACTION});
  auto work = gate.Wait(std::stop_token{});
  ASSERT_TRUE(work.has_value());
  ASSERT_TRUE(gate.Commit(*work));
  gate.PushState({LOCKSTEP_MAGIC, 4, 1, LOCKSTEP_NO_ACTION});
  work = gate.Wait(std::stop_token{});
  ASSERT_TRUE(work.has_value());
  ASSERT_TRUE(gate.Commit(*work));
  EXPECT_FALSE(gate.LastAction().has_value());
  const auto token = Token(4, 2, 0);
  EXPECT_EQ(gate.PushState(token.envelope), SimLockstepGate::PushResult::Accepted);
  EXPECT_EQ(gate.PushToken(token), SimLockstepGate::PushResult::Accepted);
}

TEST(SimLockstep, CommitReleasePublishesPriorCommandWrites) {
  SimLockstepGate gate;
  gate.PushState({LOCKSTEP_MAGIC, 3, 0, LOCKSTEP_NO_ACTION});
  auto work = gate.Wait(std::stop_token{});
  ASSERT_TRUE(work.has_value());
  int command_epoch = 0;
  auto reader = std::async(std::launch::async, [&] {
    while (!gate.LoadCommitted()) std::this_thread::yield();
    return command_epoch;
  });
  command_epoch = 42;
  ASSERT_TRUE(gate.Commit(*work));
  EXPECT_EQ(reader.get(), 42);
}

TEST(SimLockstep, StopTokenWakesWaiter) {
  SimLockstepGate gate;
  std::promise<bool> done;
  std::jthread waiter([&](std::stop_token stop) {
    done.set_value(!gate.Wait(stop).has_value());
  });
  waiter.request_stop();
  gate.Notify();
  EXPECT_TRUE(done.get_future().get());
}

TEST(StateLoggerLockstep, StrictStrideAndResetKeepGlobalIndex) {
  StateLogger logger("", 16, 29, 29, 0.02, false, {}, true);
  EXPECT_EQ(LogOne(logger), 0u);
  EXPECT_NO_THROW(logger.GetLatest(1, 0.04));
  EXPECT_THROW(logger.GetLatest(1, 0.03), std::runtime_error);
  EXPECT_EQ(logger.size(), 1u);
  logger.ResetHistoryKeepIndex();
  EXPECT_EQ(logger.size(), 0u);
  EXPECT_EQ(LogOne(logger), 1u);
}
