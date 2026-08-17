#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

inline constexpr uint32_t LOCKSTEP_MAGIC = 0x534C4B31u;
inline constexpr uint32_t LOCKSTEP_NO_ACTION = 0xFFFFFFFFu;

struct LockstepEnvelope {
  uint32_t magic = 0;
  uint32_t session = 0;
  uint32_t sim_step = 0;
  uint32_t action_seq = LOCKSTEP_NO_ACTION;

  bool operator==(const LockstepEnvelope&) const = default;
  bool IsReady() const {
    return magic == LOCKSTEP_MAGIC && session != 0 && sim_step == 0 &&
           action_seq == LOCKSTEP_NO_ACTION;
  }
  std::array<uint32_t, 4> Reserve() const {
    return {magic, session, sim_step, action_seq};
  }
};

struct LockstepTokenBundle {
  LockstepEnvelope envelope;
  int64_t frame_index = -1;
  std::vector<double> token;
  std::array<double, 7> left_hand{};
  std::array<double, 7> right_hand{};
};

struct CommittedAck {
  LockstepEnvelope envelope;
  uint64_t commit_index = 0;
};

class SimLockstepGate {
 public:
  enum class WorkKind { SessionReady, Control };
  enum class PushResult { Accepted, Duplicate, Rejected };

  struct Work {
    WorkKind kind;
    LockstepEnvelope envelope;
    std::optional<LockstepTokenBundle> token;
  };

  using EventCallback = std::function<void(const std::string&, const LockstepEnvelope&)>;

  explicit SimLockstepGate(EventCallback callback = {})
      : callback_(std::move(callback)) {}

  PushResult PushState(const LockstepEnvelope& envelope) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (envelope.magic != LOCKSTEP_MAGIC || envelope.session == 0) {
      EmitLocked("reject_state_header", envelope);
      return PushResult::Rejected;
    }
    if (!active_session_.has_value() || envelope.session != *active_session_) {
      if (!envelope.IsReady()) {
        EmitLocked("reject_new_session_without_ready", envelope);
        return PushResult::Rejected;
      }
      active_session_ = envelope.session;
      last_step_ = 0;
      last_action_ = std::nullopt;
      pending_state_.reset();
      pending_token_.reset();
      in_flight_.reset();
      ready_pending_ = envelope;
      EmitLocked("session_ready_pending", envelope);
      cv_.notify_all();
      return PushResult::Accepted;
    }
    if (envelope.IsReady()) {
      EmitLocked("duplicate_session_ready", envelope);
      return PushResult::Duplicate;
    }
    if (envelope.sim_step <= last_step_ ||
        (in_flight_.has_value() && envelope == in_flight_->envelope)) {
      EmitLocked("duplicate_state", envelope);
      return PushResult::Duplicate;
    }
    if (envelope.sim_step != last_step_ + 1) {
      EmitLocked("reject_state_step", envelope);
      return PushResult::Rejected;
    }
    if (envelope.action_seq != LOCKSTEP_NO_ACTION) {
      const uint32_t expected_action = last_action_.has_value() ? *last_action_ + 1 : 0;
      if (envelope.action_seq != expected_action) {
        EmitLocked("reject_state_action", envelope);
        return PushResult::Rejected;
      }
    }
    pending_state_ = envelope;
    EmitLocked("state_pending", envelope);
    cv_.notify_all();
    return PushResult::Accepted;
  }

  PushResult PushToken(const LockstepTokenBundle& bundle) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& envelope = bundle.envelope;
    if (envelope.magic != LOCKSTEP_MAGIC || !active_session_.has_value() ||
        envelope.session != *active_session_) {
      EmitLocked("reject_token_session", envelope);
      return PushResult::Rejected;
    }
    if (envelope.sim_step <= last_step_ ||
        (in_flight_.has_value() && envelope == in_flight_->envelope)) {
      EmitLocked("duplicate_token", envelope);
      return PushResult::Duplicate;
    }
    const uint32_t expected_action = last_action_.has_value() ? *last_action_ + 1 : 0;
    if (envelope.sim_step != last_step_ + 1 ||
        envelope.action_seq != expected_action ||
        envelope.action_seq == LOCKSTEP_NO_ACTION ||
        bundle.frame_index != static_cast<int64_t>(envelope.action_seq) ||
        bundle.token.size() != 64) {
      EmitLocked("reject_token_sequence_or_shape", envelope);
      return PushResult::Rejected;
    }
    pending_token_ = bundle;
    EmitLocked("token_pending", envelope);
    cv_.notify_all();
    return PushResult::Accepted;
  }

  std::optional<Work> Wait(std::stop_token stop) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::stop_callback wake(stop, [this] { cv_.notify_all(); });
    cv_.wait(lock, [&] { return stop.stop_requested() || HasWorkLocked(); });
    if (stop.stop_requested()) return std::nullopt;
    if (ready_pending_.has_value()) {
      Work work{WorkKind::SessionReady, *ready_pending_, std::nullopt};
      ready_pending_.reset();
      in_flight_ = work;
      return work;
    }
    if (!pending_state_.has_value()) return std::nullopt;
    const auto envelope = *pending_state_;
    if (envelope.action_seq == LOCKSTEP_NO_ACTION) {
      Work work{WorkKind::Control, envelope, std::nullopt};
      pending_state_.reset();
      in_flight_ = work;
      return work;
    }
    if (pending_token_.has_value() && pending_token_->envelope == envelope) {
      Work work{WorkKind::Control, envelope, pending_token_};
      pending_state_.reset();
      pending_token_.reset();
      in_flight_ = work;
      return work;
    }
    return std::nullopt;
  }

  bool Commit(const Work& work) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!in_flight_.has_value() || in_flight_->kind != work.kind ||
        in_flight_->envelope != work.envelope) {
      return false;
    }
    if (work.kind == WorkKind::Control) {
      last_step_ = work.envelope.sim_step;
      if (work.envelope.action_seq != LOCKSTEP_NO_ACTION) {
        last_action_ = work.envelope.action_seq;
      }
    }
    ++commit_index_;
    auto committed = std::make_shared<const CommittedAck>(
        CommittedAck{work.envelope, commit_index_});
    std::atomic_store_explicit(&committed_, std::move(committed),
                               std::memory_order_release);
    in_flight_.reset();
    EmitLocked("commit", work.envelope);
    return true;
  }

  std::shared_ptr<const CommittedAck> LoadCommitted() const {
    return std::atomic_load_explicit(&committed_, std::memory_order_acquire);
  }

  void Notify() { cv_.notify_all(); }

  uint32_t LastStep() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_step_;
  }

  std::optional<uint32_t> LastAction() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_action_;
  }

 private:
  bool HasWorkLocked() const {
    if (ready_pending_.has_value()) return true;
    if (!pending_state_.has_value()) return false;
    if (pending_state_->action_seq == LOCKSTEP_NO_ACTION) return true;
    return pending_token_.has_value() &&
           pending_token_->envelope == *pending_state_;
  }

  void EmitLocked(const std::string& event, const LockstepEnvelope& envelope) {
    if (callback_) callback_(event, envelope);
  }

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  EventCallback callback_;
  std::optional<uint32_t> active_session_;
  uint32_t last_step_ = 0;
  std::optional<uint32_t> last_action_;
  std::optional<LockstepEnvelope> ready_pending_;
  std::optional<LockstepEnvelope> pending_state_;
  std::optional<LockstepTokenBundle> pending_token_;
  std::optional<Work> in_flight_;
  uint64_t commit_index_ = 0;
  mutable std::shared_ptr<const CommittedAck> committed_;
};
